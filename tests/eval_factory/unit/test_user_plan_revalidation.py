from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_job_store import _work_graph
from test_task_rewrite import (
    REQUIREMENT_ID,
    _adjusted_plan,
    _pass_prompt_safety,
    _source_chain,
)
from test_task_rewrite import (
    _audit as _r4_audit,
)
from test_user_approval_requests import (
    _compile as _compile_request,
)
from test_user_approval_requests import (
    _environment_source,
    _final_source,
    _generation_policy,
    _job_spec,
    _policy,
)
from test_user_decisions import (
    _audit,
    _authentication,
    _compile_decision,
    _handling_policy,
    _label_request,
)

from eval_factory.approval.adjustments import (
    EnvironmentStrategyAdjustmentCompiler,
    FinalDatasetAdjustmentCompiler,
    LabelPlanAdjustmentCompiler,
)
from eval_factory.approval.decisions import (
    UserDecisionCompiler,
    task_rewrite_adjustment_effect_from_result,
)
from eval_factory.approval.requests import (
    TaskRewriteApprovalSource,
    UserApprovalRequestCompiler,
)
from eval_factory.approval.revalidation import (
    DirectedRevalidationCompiler,
    RevalidationItemSource,
    RevalidationStageSource,
    TaskRewriteApplicationSource,
    UserPlanApplicationCompiler,
    UserPlanApplicationPolicyError,
)
from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    ApprovalMode,
    EnvironmentScopeDecision,
    EnvironmentStrategyChoice,
    FinalReviewScope,
    QueryPackagingChoice,
    TypedAdjustment,
    UserDecision,
)
from eval_factory.contracts.approval_application_v2 import (
    DirectedRevalidationReportOutcomeV2,
    RevalidationWorkOutcomeV2,
    UserPlanApplicationPolicyV2,
    revalidation_work_item_v2_ref,
)
from eval_factory.contracts.approval_v2 import user_approval_request_ref
from eval_factory.contracts.core import FailureClass, FailureRecord, ObjectRef
from eval_factory.contracts.orchestration import StageRunStatus
from eval_factory.contracts.orchestration_v2 import StageNameV2
from eval_factory.orchestration.models import (
    StageResultRecord,
    StageRunRecord,
    stage_result_record_carried_sha256,
)
from eval_factory.task_authoring import (
    FakeTaskRewritePromptFixture,
    FakeTaskRewritePromptRunner,
    TaskRewriteApplicationCompiler,
    TaskRewritePromptCompiler,
    TaskRewritePromptOutcome,
    TaskRewritePromptRequestBuilder,
)

NOW = datetime(2026, 8, 2, tzinfo=UTC)
HASH = "a" * 64
ROOT = Path(__file__).resolve().parents[3]
GOLD_PATH = ROOT / "evals/golden/eval_factory/approval" / "r7-06-directed-revalidation-v1.json"


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r7-06/{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


def _application_policy(
    *,
    max_work_items: int = 100,
) -> UserPlanApplicationPolicyV2:
    return UserPlanApplicationPolicyV2.create(
        max_adjustments=10,
        max_invalidated_refs=200,
        max_preserved_refs=200,
        max_affected_items=100,
        max_work_items=max_work_items,
        max_outputs_per_work=20,
        max_revalidation_attempts=3,
        max_failure_code_characters=128,
        audit=_audit(),
    )


def _label_application_sources():
    approval_policy, approval_source, request_result, request = _label_request()
    adjustment = TypedAdjustment(
        target_path="label_plan.boundary",
        operation="SET",
        value="Use only current executed tool facts.",
        reason="Tighten the label boundary.",
    )
    producer = LabelPlanAdjustmentCompiler().compile(
        request=request,
        source_label_spec=approval_source.label_spec,
        source_label_plan=approval_source.label_plan,
        adjustments=(adjustment,),
        audit=_audit(),
    )
    decision = _compile_decision(
        policy=approval_policy,
        sources=(approval_source,),
        request_result=request_result,
        request=request,
        decision=UserDecision.ADJUST,
        adjustments=(adjustment,),
        adjustment_effect=producer.effect,
    )
    job_spec = _job_spec(approval_policy)
    graph = _work_graph(job_spec)
    item_id = graph.item_ids[0]
    trace_ref = graph.source_trace_refs[0]
    item_source = RevalidationItemSource(
        item_id=item_id,
        source_trace_ref=trace_ref,
        authority_refs=(request.subject_refs[0],),
        stages=(
            RevalidationStageSource(
                stage=StageNameV2.LABEL,
                output_refs=(_ref("label-decision", "source"),),
            ),
            RevalidationStageSource(
                stage=StageNameV2.TASK_AUTHORING,
                output_refs=(_ref("r4-task-contract-set", "source"),),
            ),
            RevalidationStageSource(
                stage=StageNameV2.ATTACHMENT,
                output_refs=(_ref("attachment-reconstruction-result", "source"),),
            ),
            RevalidationStageSource(
                stage=StageNameV2.ITEM_QUALITY,
                output_refs=(_ref("item-quality-compilation-result", "source"),),
            ),
        ),
    )
    return (
        approval_policy,
        producer,
        decision,
        job_spec,
        graph,
        (item_source,),
    )


def test_label_application_derives_closure_dag_and_checkpoint_reapproval() -> None:
    (
        _,
        producer,
        decision,
        job_spec,
        graph,
        item_sources,
    ) = _label_application_sources()
    graph_json = graph.model_dump_json()

    compiled = UserPlanApplicationCompiler().compile(
        decision_commit=decision,
        producer_result=producer.result,
        job_spec=job_spec,
        resolved_job_work_graph=graph,
        item_sources=item_sources,
        policy=_application_policy(),
        audit=_audit(),
    )

    assert graph.model_dump_json() == graph_json
    assert compiled.application.checkpoint is ApprovalCheckpoint.LABEL_PLAN
    assert compiled.application.affected_item_ids == graph.item_ids
    assert compiled.application.required_checkpoint_reapprovals == (
        ApprovalCheckpoint.TASK_REWRITE_PLAN,
        ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
    )
    assert compiled.application.final_checkpoint_reapproval_required is False
    assert tuple(work.stage for work in compiled.plan.work_items) == (
        StageNameV2.LABEL,
        StageNameV2.TASK_AUTHORING,
        StageNameV2.ATTACHMENT,
        StageNameV2.ITEM_QUALITY,
        StageNameV2.BATCH_QUALITY,
    )
    assert all(work.stage is not StageNameV2.RELEASE for work in compiled.plan.work_items)
    assert compiled.plan.preserved_item_ids == ()
    UserPlanApplicationCompiler().validate_current(
        compiled,
        decision_commit=decision,
        producer_result=producer.result,
        job_spec=job_spec,
        resolved_job_work_graph=graph,
        item_sources=item_sources,
        policy=_application_policy(),
    )


def test_application_rejects_non_adjust_and_work_budget() -> None:
    (
        approval_policy,
        producer,
        decision,
        job_spec,
        graph,
        item_sources,
    ) = _label_application_sources()
    _, source, request_result, request = _label_request()
    accepted = _compile_decision(
        policy=approval_policy,
        sources=(source,),
        request_result=request_result,
        request=request,
        decision=UserDecision.ACCEPT,
    )

    with pytest.raises(UserPlanApplicationPolicyError, match="ADJUSTMENT_ACCEPTED"):
        UserPlanApplicationCompiler().compile(
            decision_commit=accepted,
            producer_result=producer.result,
            job_spec=job_spec,
            resolved_job_work_graph=graph,
            item_sources=item_sources,
            policy=_application_policy(),
            audit=_audit(),
        )
    with pytest.raises(UserPlanApplicationPolicyError, match="work item limit"):
        UserPlanApplicationCompiler().compile(
            decision_commit=decision,
            producer_result=producer.result,
            job_spec=job_spec,
            resolved_job_work_graph=graph,
            item_sources=item_sources,
            policy=_application_policy(max_work_items=2),
            audit=_audit(),
        )


@pytest.mark.parametrize(
    ("directive", "expected_stages", "excluded"),
    (
        (
            "REBUILD_FROM_TASK_AUTHORING",
            (
                StageNameV2.TASK_AUTHORING,
                StageNameV2.ATTACHMENT,
                StageNameV2.ITEM_QUALITY,
                StageNameV2.BATCH_QUALITY,
            ),
            False,
        ),
        (
            "REBUILD_FROM_ATTACHMENT",
            (
                StageNameV2.ATTACHMENT,
                StageNameV2.ITEM_QUALITY,
                StageNameV2.BATCH_QUALITY,
            ),
            False,
        ),
        (
            "REBUILD_FROM_ITEM_QUALITY",
            (
                StageNameV2.ITEM_QUALITY,
                StageNameV2.BATCH_QUALITY,
            ),
            False,
        ),
        (
            "EXCLUDE_ITEM",
            (StageNameV2.BATCH_QUALITY,),
            True,
        ),
    ),
)
def test_final_adjustment_requires_current_final_reapproval(
    directive: str,
    expected_stages: tuple[StageNameV2, ...],
    excluded: bool,
) -> None:
    policy = _policy(
        ApprovalMode.FINAL_ONLY,
        scope=FinalReviewScope.SELECTED_ITEMS,
    )
    source = _final_source(FinalReviewScope.SELECTED_ITEMS)
    request_result = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.FINAL_DATASET_REVIEW,
        sources=(source,),
    )
    request = request_result.requests[0]
    target_ref = source.preview.subject_refs[0]
    adjustment = TypedAdjustment(
        target_path=(f"items[{hashlib.sha256(target_ref.object_id.encode()).hexdigest()}].restart_stage"),
        operation="SET",
        value=directive,
        reason="Re-run quality for the selected Item.",
    )
    producer = FinalDatasetAdjustmentCompiler().compile(
        request=request,
        source_preview=source.preview,
        adjustments=(adjustment,),
        audit=_audit(),
    )
    decision = _compile_decision(
        policy=policy,
        sources=(source,),
        request_result=request_result,
        request=request,
        decision=UserDecision.ADJUST,
        adjustments=(adjustment,),
        adjustment_effect=producer.effect,
    )
    job_spec = _job_spec(policy)
    graph = _work_graph(job_spec)
    item_id = graph.item_ids[0]
    item_sources = (
        RevalidationItemSource(
            item_id=item_id,
            source_trace_ref=graph.source_trace_refs[0],
            authority_refs=(target_ref,),
            stages=(
                RevalidationStageSource(
                    stage=StageNameV2.ITEM_QUALITY,
                    output_refs=(
                        _ref(
                            "item-quality-compilation-result",
                            "source-final",
                        ),
                    ),
                ),
            ),
        ),
    )

    compiled = UserPlanApplicationCompiler().compile(
        decision_commit=decision,
        producer_result=producer.result,
        job_spec=job_spec,
        resolved_job_work_graph=graph,
        item_sources=item_sources,
        policy=_application_policy(),
        audit=_audit(),
    )

    assert compiled.application.required_checkpoint_reapprovals == (ApprovalCheckpoint.FINAL_DATASET_REVIEW,)
    assert compiled.application.final_checkpoint_reapproval_required is True
    assert tuple(work.stage for work in compiled.plan.work_items) == expected_stages
    assert compiled.application.excluded_item_ids == ((item_id,) if excluded else ())


def test_final_adjustment_preserves_unaffected_item_authority() -> None:
    policy = _policy(
        ApprovalMode.FINAL_ONLY,
        scope=FinalReviewScope.SELECTED_ITEMS,
    )
    base_job = _job_spec(policy)
    second_trace = base_job.traces[0].model_copy(
        update={
            "source_trace_id": "source-trace://r7-06/unaffected",
            "source_uri": "raw-traj://r7-06/unaffected",
            "raw_sha256": "b" * 64,
        }
    )
    job_spec = base_job.model_copy(update={"traces": (*base_job.traces, second_trace)})
    source = _final_source(FinalReviewScope.SELECTED_ITEMS)
    request_result = UserApprovalRequestCompiler().compile(
        job_spec=job_spec,
        approval_policy=policy,
        generation_policy=_generation_policy(),
        checkpoint=ApprovalCheckpoint.FINAL_DATASET_REVIEW,
        sources=(source,),
        requested_by="requesting-user",
        audit=_audit(),
    )
    request = request_result.requests[0]
    target_ref = source.preview.subject_refs[0]
    adjustment = TypedAdjustment(
        target_path=(f"items[{hashlib.sha256(target_ref.object_id.encode()).hexdigest()}].restart_stage"),
        operation="SET",
        value="REBUILD_FROM_ITEM_QUALITY",
        reason="Re-run only the selected Item.",
    )
    producer = FinalDatasetAdjustmentCompiler().compile(
        request=request,
        source_preview=source.preview,
        adjustments=(adjustment,),
        audit=_audit(),
    )
    decision = UserDecisionCompiler().compile(
        job_spec=job_spec,
        approval_policy=policy,
        generation_policy=_generation_policy(),
        request_compilation=request_result,
        checkpoint_sources=(source,),
        request_ref=user_approval_request_ref(request),
        handling_policy=_handling_policy(),
        authentication=_authentication(),
        decision=UserDecision.ADJUST,
        adjustments=(adjustment,),
        adjustment_effect=producer.effect,
        environment_decisions=(),
        query_packaging=None,
        reason="Apply targeted final-review invalidation.",
        idempotency_key="r7-06-final-target-only",
        decided_at=NOW,
        audit=_audit(),
    )
    graph = _work_graph(job_spec)
    target_quality_ref = _ref(
        "item-quality-compilation-result",
        "target-current",
    )
    preserved_quality_ref = _ref(
        "item-quality-compilation-result",
        "unaffected-current",
    )
    item_sources = (
        RevalidationItemSource(
            item_id=graph.item_ids[0],
            source_trace_ref=graph.source_trace_refs[0],
            authority_refs=(target_ref,),
            stages=(
                RevalidationStageSource(
                    stage=StageNameV2.ITEM_QUALITY,
                    output_refs=(target_quality_ref,),
                ),
            ),
        ),
        RevalidationItemSource(
            item_id=graph.item_ids[1],
            source_trace_ref=graph.source_trace_refs[1],
            authority_refs=(_ref("evaluation-item", "unaffected"),),
            stages=(
                RevalidationStageSource(
                    stage=StageNameV2.ITEM_QUALITY,
                    output_refs=(preserved_quality_ref,),
                ),
            ),
        ),
    )

    compiled = UserPlanApplicationCompiler().compile(
        decision_commit=decision,
        producer_result=producer.result,
        job_spec=job_spec,
        resolved_job_work_graph=graph,
        item_sources=item_sources,
        policy=_application_policy(),
        audit=_audit(),
    )

    assert compiled.application.affected_item_ids == (graph.item_ids[0],)
    assert compiled.plan.preserved_item_ids == (graph.item_ids[1],)
    assert preserved_quality_ref in compiled.application.preserved_object_refs
    assert preserved_quality_ref not in (compiled.application.invalidated_object_refs)
    assert {work.item_id for work in compiled.plan.work_items if work.item_id is not None} == {
        graph.item_ids[0]
    }


@pytest.mark.parametrize(
    ("strategy", "expected_stages", "excluded"),
    (
        (
            EnvironmentStrategyChoice.TRACE_FAITHFUL_MOCK,
            (
                StageNameV2.ATTACHMENT,
                StageNameV2.ITEM_QUALITY,
                StageNameV2.BATCH_QUALITY,
            ),
            False,
        ),
        (
            EnvironmentStrategyChoice.REWRITE_STANDARD_ENV,
            (
                StageNameV2.TASK_AUTHORING,
                StageNameV2.ATTACHMENT,
                StageNameV2.ITEM_QUALITY,
                StageNameV2.BATCH_QUALITY,
            ),
            False,
        ),
        (
            EnvironmentStrategyChoice.EXCLUDE_TASK,
            (StageNameV2.BATCH_QUALITY,),
            True,
        ),
    ),
)
def test_environment_strategy_derives_item_restart_scope(
    strategy: EnvironmentStrategyChoice,
    expected_stages: tuple[StageNameV2, ...],
    excluded: bool,
) -> None:
    source = _environment_source()
    policy = _policy(ApprovalMode.PLAN_GATES)
    request_result = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        sources=(source,),
    )
    request = request_result.requests[0]
    requirement = source.strategy.requirements[0]
    adjustment = TypedAdjustment(
        target_path=(
            f"requirements[{hashlib.sha256(requirement.requirement_id.encode()).hexdigest()}].description"
        ),
        operation="SET",
        value="Use a reproducible local workspace.",
        reason="Clarify the environment requirement.",
    )
    producer = EnvironmentStrategyAdjustmentCompiler().compile(
        request=request,
        source_strategy=source.strategy,
        adjustments=(adjustment,),
        audit=_audit(),
    )
    decision = _compile_decision(
        policy=policy,
        sources=(source,),
        request_result=request_result,
        request=request,
        decision=UserDecision.ADJUST,
        adjustments=(adjustment,),
        adjustment_effect=producer.effect,
        environment_decisions=(
            EnvironmentScopeDecision(
                requirement_id=requirement.requirement_id,
                strategy=strategy,
            ),
        ),
        query_packaging=QueryPackagingChoice.INCLUDE_QUERY_YAML,
    )
    job_spec = _job_spec(policy)
    graph = _work_graph(job_spec)
    item_id = graph.item_ids[0]
    item_sources = (
        RevalidationItemSource(
            item_id=item_id,
            source_trace_ref=graph.source_trace_refs[0],
            authority_refs=requirement.affected_subject_refs,
            environment_requirement_ids=(requirement.requirement_id,),
            stages=(
                RevalidationStageSource(
                    stage=StageNameV2.TASK_AUTHORING,
                    output_refs=(_ref("r4-task-contract-set", "environment-source"),),
                ),
                RevalidationStageSource(
                    stage=StageNameV2.ATTACHMENT,
                    output_refs=(
                        _ref(
                            "attachment-reconstruction-result",
                            "environment-source",
                        ),
                    ),
                ),
                RevalidationStageSource(
                    stage=StageNameV2.ITEM_QUALITY,
                    output_refs=(
                        _ref(
                            "item-quality-compilation-result",
                            "environment-source",
                        ),
                    ),
                ),
            ),
        ),
    )

    compiled = UserPlanApplicationCompiler().compile(
        decision_commit=decision,
        producer_result=producer.result,
        job_spec=job_spec,
        resolved_job_work_graph=graph,
        item_sources=item_sources,
        policy=_application_policy(),
        audit=_audit(),
    )

    assert tuple(work.stage for work in compiled.plan.work_items) == expected_stages
    assert compiled.application.excluded_item_ids == ((item_id,) if excluded else ())


def test_task_rewrite_application_reuses_complete_r4_authority() -> None:
    chain = _source_chain()
    source_contract_set, initial, adjusted = _adjusted_plan(chain)
    assert initial.plan_version is not None
    assert initial.preview_safety_gate is not None
    assert initial.preview is not None
    assert adjusted.replacement_plan_version is not None
    assert adjusted.replacement_preview_safety_gate is not None
    assert adjusted.replacement_preview is not None
    assert adjusted.invalidation is not None

    prompt_request = TaskRewritePromptRequestBuilder().build(
        source_task_draft=chain["task_draft"],
        replacement_plan_version=adjusted.replacement_plan_version,
        replacement_preview=adjusted.replacement_preview,
        replacement_preview_safety_gate=(adjusted.replacement_preview_safety_gate),
        model_profile="internal-task-rewriter-v1",
        prompt_version="task-rewrite-prompt/v1",
        audit=_r4_audit(),
    )
    prompt_proposal = FakeTaskRewritePromptRunner().run(
        prompt_request,
        fixture=FakeTaskRewritePromptFixture(
            fixture_id="fixture://r7-06/task-rewrite",
            outcome=TaskRewritePromptOutcome.REWRITTEN,
            visible_prompt=("Inspect the input-state workspace and summarize its design."),
            covered_prompt_requirement_ids=(REQUIREMENT_ID,),
        ),
        audit=_r4_audit(),
    )
    rewritten = TaskRewritePromptCompiler().compile(
        request=prompt_request,
        proposal=prompt_proposal,
        source_task_draft=chain["task_draft"],
        replacement_plan_version=adjusted.replacement_plan_version,
        replacement_preview=adjusted.replacement_preview,
        replacement_preview_safety_gate=(adjusted.replacement_preview_safety_gate),
        audit=_r4_audit(),
    )
    assert rewritten.task_draft is not None
    passed_task, passed_gate, _ = _pass_prompt_safety(
        rewritten.task_draft,
        reference_set=chain["reference_set"],
    )
    application_result = TaskRewriteApplicationCompiler().compile(
        source_contract_set=source_contract_set,
        source_plan_version=adjusted.source_plan_version,
        replacement_plan_version=adjusted.replacement_plan_version,
        invalidation=adjusted.invalidation,
        source_task_draft=chain["task_draft"],
        source_task_prompt_safety_gate=chain["gate"],
        source_rubric_set=chain["rubric_set"],
        source_evaluator_spec=chain["evaluator_spec"],
        source_reference_policy=chain["reference_policy"],
        source_tool_policy=chain["tool_policy"],
        source_contestant_tool_policy=chain["contestant_policy"],
        source_storage_authorization=chain["storage_authorization"],
        source_producer_task_view=chain["producer_task_view"],
        rewritten_pending_task_draft=rewritten.task_draft,
        rewritten_passed_task_draft=passed_task,
        rewritten_task_prompt_safety_gate=passed_gate,
        tool_catalog=chain["catalog"],
        producer_view_result=chain["producer_view_result"],
        producer_evidence_bundle=chain["producer_bundle"],
        audit=_r4_audit(),
    )

    adjustment = TypedAdjustment(
        target_path="rewrite_style",
        operation="SET",
        value="Direct and concise.",
        reason="Use the configured task style.",
    )
    policy = _policy(ApprovalMode.PLAN_GATES)
    approval_source = TaskRewriteApprovalSource(
        plan_version=initial.plan_version,
        preview_safety_gate=initial.preview_safety_gate,
        preview=initial.preview,
    )
    request_result = _compile_request(
        policy=policy,
        checkpoint=ApprovalCheckpoint.TASK_REWRITE_PLAN,
        sources=(approval_source,),
    )
    request = request_result.requests[0]
    effect = task_rewrite_adjustment_effect_from_result(
        request=request,
        adjustments=(adjustment,),
        result=adjusted,
        audit=_audit(),
    )
    decision = _compile_decision(
        policy=policy,
        sources=(approval_source,),
        request_result=request_result,
        request=request,
        decision=UserDecision.ADJUST,
        adjustments=(adjustment,),
        adjustment_effect=effect,
    )
    job_spec = _job_spec(policy)
    graph = _work_graph(job_spec)
    item_sources = (
        RevalidationItemSource(
            item_id=graph.item_ids[0],
            source_trace_ref=graph.source_trace_refs[0],
            authority_refs=(request.subject_refs[0],),
            stages=(
                RevalidationStageSource(
                    stage=StageNameV2.TASK_AUTHORING,
                    output_refs=(_ref("r4-task-contract-set", "rewrite-source"),),
                ),
                RevalidationStageSource(
                    stage=StageNameV2.ATTACHMENT,
                    output_refs=(
                        _ref(
                            "attachment-reconstruction-result",
                            "rewrite-source",
                        ),
                    ),
                ),
                RevalidationStageSource(
                    stage=StageNameV2.ITEM_QUALITY,
                    output_refs=(
                        _ref(
                            "item-quality-compilation-result",
                            "rewrite-source",
                        ),
                    ),
                ),
            ),
        ),
    )

    compiled = UserPlanApplicationCompiler().compile(
        decision_commit=decision,
        producer_result=TaskRewriteApplicationSource(
            adjustment_result=adjusted,
            application_result=application_result,
        ),
        job_spec=job_spec,
        resolved_job_work_graph=graph,
        item_sources=item_sources,
        policy=_application_policy(),
        audit=_audit(),
    )

    assert tuple(work.stage for work in compiled.plan.work_items) == (
        StageNameV2.TASK_AUTHORING,
        StageNameV2.ATTACHMENT,
        StageNameV2.ITEM_QUALITY,
        StageNameV2.BATCH_QUALITY,
    )
    assert compiled.application.required_checkpoint_reapprovals == (ApprovalCheckpoint.ENVIRONMENT_STRATEGY,)
    assert application_result.application.replacement_contract_set_ref in (
        compiled.application.replacement_root_refs
    )
    stale_plan = adjusted.replacement_plan_version.plan.model_copy(
        update={"target_capability": "Caller-mutated stale authority."}
    )
    stale_adjustment = adjusted.model_copy(
        update={
            "replacement_plan_version": adjusted.replacement_plan_version.model_copy(
                update={"plan": stale_plan}
            )
        }
    )
    with pytest.raises(
        UserPlanApplicationPolicyError,
        match="replacement plan version is stale",
    ):
        UserPlanApplicationCompiler().compile(
            decision_commit=decision,
            producer_result=TaskRewriteApplicationSource(
                adjustment_result=stale_adjustment,
                application_result=application_result,
            ),
            job_spec=job_spec,
            resolved_job_work_graph=graph,
            item_sources=item_sources,
            policy=_application_policy(),
            audit=_audit(),
        )


def _stage_witness(
    *,
    work_ref: ObjectRef,
    stage: StageNameV2,
    item_id: str | None,
    output_type: str,
) -> tuple[StageRunRecord, StageResultRecord]:
    run = StageRunRecord(
        stage_run_id=f"stage-run://r7-06/{stage.value}/{item_id or 'job'}",
        job_id="job://r7-04/requests",
        item_id=item_id,
        stage=stage,
        attempt=1,
        status=StageRunStatus.SUCCEEDED,
        principal_ref=_ref("principal", "worker"),
        input_refs=(work_ref,),
        row_version=2,
        idempotency_key=f"run-{stage.value}-{item_id or 'job'}",
        created_at=NOW,
        started_at=NOW,
        ended_at=NOW,
    )
    result = StageResultRecord(
        stage_result_id=f"stage-result://r7-06/{stage.value}/{item_id or 'job'}",
        stage_run_id=run.stage_run_id,
        stage_run_version=run.row_version,
        status=StageRunStatus.SUCCEEDED,
        output_refs=(_ref(output_type, f"replacement-{stage.value}"),),
        result_sha256="0" * 64,
        created_at=NOW,
    )
    digest = stage_result_record_carried_sha256(result)
    return run, result.model_copy(update={"result_sha256": digest})


def _stage_failure_witness(
    *,
    work_ref: ObjectRef,
    stage: StageNameV2,
    item_id: str | None,
    status: StageRunStatus,
) -> tuple[StageRunRecord, StageResultRecord]:
    run = StageRunRecord(
        stage_run_id=(f"stage-run://r7-06/{stage.value}/{item_id or 'job'}/{status.value}"),
        job_id="job://r7-04/requests",
        item_id=item_id,
        stage=stage,
        attempt=1,
        status=status,
        principal_ref=_ref("principal", "worker"),
        input_refs=(work_ref,),
        row_version=2,
        idempotency_key=(f"run-{stage.value}-{item_id or 'job'}-{status.value}"),
        created_at=NOW,
        started_at=NOW,
        ended_at=NOW,
    )
    failure_class = {
        StageRunStatus.BLOCKED_POLICY: FailureClass.POLICY,
        StageRunStatus.BLOCKED_CAPABILITY: FailureClass.CAPABILITY,
        StageRunStatus.TERMINAL_FAILURE: FailureClass.INTERNAL,
    }[status]
    result = StageResultRecord(
        stage_result_id=(f"stage-result://r7-06/{stage.value}/{item_id or 'job'}/{status.value}"),
        stage_run_id=run.stage_run_id,
        stage_run_version=run.row_version,
        status=status,
        failure=FailureRecord(
            failure_class=failure_class,
            code=f"R7_06_{status.value}",
            message="Closed revalidation failure.",
            retryable=False,
        ),
        result_sha256="0" * 64,
        created_at=NOW,
    )
    digest = stage_result_record_carried_sha256(result)
    return run, result.model_copy(update={"result_sha256": digest})


@pytest.mark.parametrize(
    ("status", "expected_outcome"),
    (
        (
            StageRunStatus.BLOCKED_POLICY,
            DirectedRevalidationReportOutcomeV2.BLOCKED,
        ),
        (
            StageRunStatus.BLOCKED_CAPABILITY,
            DirectedRevalidationReportOutcomeV2.BLOCKED,
        ),
        (
            StageRunStatus.TERMINAL_FAILURE,
            DirectedRevalidationReportOutcomeV2.FAILED,
        ),
    ),
)
def test_terminal_frontier_compiles_non_approvable_report(
    status: StageRunStatus,
    expected_outcome: DirectedRevalidationReportOutcomeV2,
) -> None:
    (
        _,
        producer,
        decision,
        job_spec,
        graph,
        item_sources,
    ) = _label_application_sources()
    compiled = UserPlanApplicationCompiler().compile(
        decision_commit=decision,
        producer_result=producer.result,
        job_spec=job_spec,
        resolved_job_work_graph=graph,
        item_sources=item_sources,
        policy=_application_policy(),
        audit=_audit(),
    )
    work = compiled.plan.work_items[0]
    run, stage_result = _stage_failure_witness(
        work_ref=revalidation_work_item_v2_ref(work),
        stage=work.stage,
        item_id=work.item_id,
        status=status,
    )
    compiler = DirectedRevalidationCompiler()
    result = compiler.record_result(
        application=compiled.application,
        plan=compiled.plan,
        work_item=work,
        stage_run=run,
        stage_result=stage_result,
        dependency_results=(),
        policy=_application_policy(),
        audit=_audit(),
    )
    report = compiler.compile_report(
        application=compiled.application,
        plan=compiled.plan,
        work_results=(result,),
        audit=_audit(),
    )

    assert report.outcome is expected_outcome
    assert report.replacement_current_refs == ()
    compiler.validate_current(
        report,
        application=compiled.application,
        plan=compiled.plan,
        work_results=(result,),
    )


def test_result_witnesses_follow_dependencies_and_complete_report() -> None:
    (
        _,
        producer,
        decision,
        job_spec,
        graph,
        item_sources,
    ) = _label_application_sources()
    compiled = UserPlanApplicationCompiler().compile(
        decision_commit=decision,
        producer_result=producer.result,
        job_spec=job_spec,
        resolved_job_work_graph=graph,
        item_sources=item_sources,
        policy=_application_policy(),
        audit=_audit(),
    )
    compiler = DirectedRevalidationCompiler()
    results = []
    for work in compiled.plan.work_items:
        run, stage_result = _stage_witness(
            work_ref=revalidation_work_item_v2_ref(work),
            stage=work.stage,
            item_id=work.item_id,
            output_type=work.required_output_types[0],
        )
        result = compiler.record_result(
            application=compiled.application,
            plan=compiled.plan,
            work_item=work,
            stage_run=run,
            stage_result=stage_result,
            dependency_results=tuple(
                value for value in results if value.work_item_ref in work.depends_on_work_item_refs
            ),
            policy=_application_policy(),
            audit=_audit(),
        )
        assert result.outcome is RevalidationWorkOutcomeV2.SUCCEEDED
        results.append(result)

    report = compiler.compile_report(
        application=compiled.application,
        plan=compiled.plan,
        work_results=tuple(results),
        audit=_audit(),
    )

    assert report.outcome is DirectedRevalidationReportOutcomeV2.COMPLETE
    assert report.replacement_current_refs
    assert report.required_checkpoint_reapprovals == (compiled.application.required_checkpoint_reapprovals)


def test_result_rejects_missing_dependency_and_wrong_output_type() -> None:
    (
        _,
        producer,
        decision,
        job_spec,
        graph,
        item_sources,
    ) = _label_application_sources()
    compiled = UserPlanApplicationCompiler().compile(
        decision_commit=decision,
        producer_result=producer.result,
        job_spec=job_spec,
        resolved_job_work_graph=graph,
        item_sources=item_sources,
        policy=_application_policy(),
        audit=_audit(),
    )
    work = compiled.plan.work_items[1]
    run, stage_result = _stage_witness(
        work_ref=revalidation_work_item_v2_ref(work),
        stage=work.stage,
        item_id=work.item_id,
        output_type="wrong-output",
    )

    with pytest.raises(UserPlanApplicationPolicyError, match="dependency"):
        DirectedRevalidationCompiler().record_result(
            application=compiled.application,
            plan=compiled.plan,
            work_item=work,
            stage_run=run,
            stage_result=stage_result,
            dependency_results=(),
            policy=_application_policy(),
            audit=_audit(),
        )

    first_work = compiled.plan.work_items[0]
    first_run, wrong_result = _stage_witness(
        work_ref=revalidation_work_item_v2_ref(first_work),
        stage=first_work.stage,
        item_id=first_work.item_id,
        output_type="wrong-output",
    )
    with pytest.raises(UserPlanApplicationPolicyError, match="output"):
        DirectedRevalidationCompiler().record_result(
            application=compiled.application,
            plan=compiled.plan,
            work_item=first_work,
            stage_run=first_run,
            stage_result=wrong_result,
            dependency_results=(),
            policy=_application_policy(),
            audit=_audit(),
        )


def test_directed_revalidation_gold_is_content_free_and_complete() -> None:
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    scenarios = {scenario["scenario_id"]: scenario for scenario in payload["scenarios"]}

    assert payload["schema_version"] == ("eval-factory/directed-revalidation-gold/r7-06-v1")
    assert payload["direct_release_authority_count"] == 0
    assert {
        "label-directed-closure",
        "task-rewrite-r4-application",
        "environment-trace-faithful",
        "environment-standard-rewrite",
        "environment-exclude",
        "final-rebuild-task",
        "final-rebuild-attachment",
        "final-rebuild-quality",
        "final-exclude",
        "unaffected-item",
        "idempotent-application",
        "divergent-application",
        "blocked-hard-gate",
        "stale-source",
    } == set(scenarios)
    assert all(
        scenario.get("direct_content_patch_allowed") is False
        for scenario in scenarios.values()
        if scenario["scenario_id"].startswith("final-")
    )
