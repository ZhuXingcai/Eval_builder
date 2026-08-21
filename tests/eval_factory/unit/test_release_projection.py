from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_batch_quality_report import _compile as _compile_batch
from test_batch_quality_report import _inputs as _batch_inputs
from test_user_approval_requests import _policy as _approval_policy

from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    ApprovalMode,
    FinalReviewScope,
    UserDecision,
    UserDecisionRecord,
)
from eval_factory.contracts.approval_application_v2 import (
    DirectedRevalidationReportOutcomeV2,
    DirectedRevalidationReportV2,
    RevalidationWorkOutcomeV2,
    RevalidationWorkResultV2,
    directed_revalidation_report_v2_ref,
)
from eval_factory.contracts.approval_decision_v2 import (
    UserDecisionCommitOutcomeV2,
    UserDecisionCommitResultV2,
    user_decision_record_carried_sha256,
)
from eval_factory.contracts.approval_v2 import user_approval_policy_ref
from eval_factory.contracts.attachment_v2 import (
    AttachmentReconstructionOutcomeV2,
    AttachmentReconstructionResultV2,
    attachment_reconstruction_result_v2_carried_sha256,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.orchestration import (
    ConcurrencyLimit,
    ExportTarget,
    ItemStatus,
    ResourceBudget,
    TraceSourceRef,
)
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    StageNameV2,
)
from eval_factory.contracts.release import ReleaseChannel
from eval_factory.contracts.release_projection_v2 import (
    ReleaseProjectionPhaseV2,
    ReleaseProjectionPolicyV2,
    evaluation_item_v2_ref,
)
from eval_factory.dataset.release import (
    EvaluationItemReleaseSource,
    ReleaseProjectionCompiler,
    ReleaseProjectionPendingError,
    ReleaseProjectionPolicyError,
)
from eval_factory.orchestration.models import ItemRecord

NOW = datetime(2026, 8, 2, tzinfo=UTC)
GOLD_PATH = (
    Path(__file__).resolve().parents[3]
    / "evals/golden/eval_factory/release"
    / "r7-08-release-projection-v1.json"
)


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r7-08/{suffix}",
        object_version=version,
        object_sha256=hashlib.sha256(f"{object_type}:{suffix}:{version}".encode()).hexdigest(),
    )


def _audit(*refs: ObjectRef) -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="release-projection-test",
        governing_versions=(
            VersionBinding(
                component="release-projection",
                version="release-projection/r7-08-v1",
            ),
        ),
        input_refs=tuple(
            sorted(
                set(refs),
                key=lambda value: (
                    value.object_type,
                    value.object_id,
                    value.object_version,
                    value.object_sha256,
                ),
            )
        ),
    )


def _policy() -> ReleaseProjectionPolicyV2:
    return ReleaseProjectionPolicyV2.create(
        max_source_trace_refs=8,
        max_label_decision_refs=32,
        max_user_decision_refs=32,
        max_revalidation_reports=32,
        max_current_head_refs=256,
        max_chain_depth=32,
        allowed_channels=frozenset(
            {
                ReleaseChannel.CANARY,
                ReleaseChannel.INTERNAL_REVIEW,
            }
        ),
        allowed_export_profiles=frozenset({"LH"}),
        audit=_audit(),
    )


def _policy_with(
    *,
    max_chain_depth: int = 32,
    max_source_trace_refs: int = 8,
    max_label_decision_refs: int = 32,
    max_current_head_refs: int = 256,
) -> ReleaseProjectionPolicyV2:
    return ReleaseProjectionPolicyV2.create(
        max_source_trace_refs=max_source_trace_refs,
        max_label_decision_refs=max_label_decision_refs,
        max_user_decision_refs=32,
        max_revalidation_reports=32,
        max_current_head_refs=max_current_head_refs,
        max_chain_depth=max_chain_depth,
        allowed_channels=frozenset(
            {
                ReleaseChannel.CANARY,
                ReleaseChannel.INTERNAL_REVIEW,
            }
        ),
        allowed_export_profiles=frozenset({"LH"}),
        audit=_audit(),
    )


def _job_spec(graph, approval_policy) -> DatasetJobSpecV2:
    stages = tuple(
        stage
        for stage in StageNameV2
        if stage
        not in {
            StageNameV2.LABEL_PLAN,
            StageNameV2.TASK_REWRITE_PLAN,
            StageNameV2.ENVIRONMENT_STRATEGY,
            StageNameV2.FINAL_DATASET_REVIEW,
        }
        or {
            StageNameV2.LABEL_PLAN: ApprovalCheckpoint.LABEL_PLAN,
            StageNameV2.TASK_REWRITE_PLAN: ApprovalCheckpoint.TASK_REWRITE_PLAN,
            StageNameV2.ENVIRONMENT_STRATEGY: (ApprovalCheckpoint.ENVIRONMENT_STRATEGY),
            StageNameV2.FINAL_DATASET_REVIEW: (ApprovalCheckpoint.FINAL_DATASET_REVIEW),
        }[stage]
        in approval_policy.enabled_checkpoints
    )
    traces = tuple(
        TraceSourceRef(
            source_trace_id=ref.object_id,
            source_uri=f"raw-traj://{index}",
            raw_sha256=ref.object_sha256,
            adapter_name="raw-traj-v1",
            adapter_version=ref.object_version,
            processing_class="RESTRICTED_TRACE_RAW",
        )
        for index, ref in enumerate(graph.source_trace_refs)
    )
    return DatasetJobSpecV2(
        job_id=graph.job_id,
        traces=traces,
        requested_stages=stages,
        privacy_profile="trusted-monitored-local",
        model_profiles=(),
        budget=ResourceBudget(
            max_model_requests=10,
            max_model_tokens=10_000,
            max_processes=2,
            max_renderers=2,
            max_network_requests=10,
            max_storage_bytes=1_000_000,
        ),
        concurrency=ConcurrencyLimit(
            model_requests=1,
            processes=1,
            renderers=1,
            network_requests=1,
            artifacts_per_item=2,
            items=2,
        ),
        selection_spec_ref=None,
        approval_policy_ref=user_approval_policy_ref(approval_policy),
        approval_mode=approval_policy.mode,
        enabled_checkpoints=approval_policy.enabled_checkpoints,
        export_target=ExportTarget(
            profile="LH",
            profile_version="v1",
            channel="CANARY",
            registry="registry://r7-08/canary",
        ),
        idempotency_key="job-r7-08-release",
        audit=_audit(),
    )


def _attachment_result(source) -> AttachmentReconstructionResultV2:
    refs = (
        _ref("artifact-routing-request", "none", version="r5-05"),
        _ref(
            "artifact-routing-compilation-result",
            "none",
            version="r5-05",
        ),
    )
    pending = AttachmentReconstructionResultV2(
        attachment_reconstruction_result_v2_id=("attachment-reconstruction-result://pending"),
        routing_request_ref=refs[0],
        routing_compilation_result_ref=refs[1],
        attachment_planning_context_ref=_ref(
            "attachment-planning-context",
            "none",
        ),
        producer_task_view_ref=source.task_contract_set.producer_task_view_ref,
        evidence_matrix_ref=None,
        artifact_routing_plan_ref=None,
        artifact_execution_plan_ref=None,
        artifact_execution_batch_ref=None,
        artifact_results=(),
        artifact_result_refs=(),
        outcome=AttachmentReconstructionOutcomeV2.NOT_REQUIRED,
        candidate_output_refs=(),
        accepted_artifact_refs=(),
        failed_artifact_ids=(),
        resumable_artifact_ids=(),
        dependency_blocked_artifact_ids=(),
        required_incomplete_artifact_ids=(),
        optional_incomplete_artifact_ids=(),
        frozen_result=None,
        frozen_projection_gap="PRE_VALIDATION",
        environment_spec_ref=None,
        provenance_manifest_ref=None,
        quality_report_ref=None,
        package_sha256=None,
        input_state_only=None,
        attachment_reconstruction_result_v2_sha256="0" * 64,
        audit=ContractAudit(
            created_at=NOW,
            created_by="release-projection-test",
            governing_versions=(
                VersionBinding(
                    component="artifact-results",
                    version="r5-07",
                ),
            ),
            input_refs=tuple(sorted(refs, key=lambda value: value.object_id)),
        ),
    )
    digest = attachment_reconstruction_result_v2_carried_sha256(pending)
    return pending.model_copy(
        update={
            "attachment_reconstruction_result_v2_id": (f"attachment-reconstruction-result://sha256/{digest}"),
            "attachment_reconstruction_result_v2_sha256": digest,
        }
    )


async def _source(
    *,
    mode: ApprovalMode,
    final_scope: FinalReviewScope = FinalReviewScope.NONE,
    item_index: int = 0,
) -> EvaluationItemReleaseSource:
    inputs = await _batch_inputs(
        prompts=(
            "Inspect alpha workspace inputs and summarize the design.",
            "Analyze beta workspace requirements and report the structure.",
        ),
        restricted_texts=(
            "alpha answer one two three four five six seven eight",
            "beta private reference nine ten eleven twelve thirteen fourteen",
        ),
    )
    graph, item_ids, sources, *_ = inputs
    batch = _compile_batch(inputs)
    approval = _approval_policy(mode, scope=final_scope)
    selected = sources[item_index]
    return EvaluationItemReleaseSource(
        job_spec=_job_spec(graph, approval),
        item=ItemRecord(
            item_id=item_ids[item_index],
            job_id=graph.job_id,
            status=ItemStatus.RUNNING,
            row_version=1,
            idempotency_key="item-r7-08-release",
            created_at=NOW,
            updated_at=NOW,
        ),
        resolved_job_work_graph=graph,
        source_trace_refs=(selected.source_trace_ref,),
        label_decisions=selected.label_decisions,
        task_draft=selected.task_draft,
        task_prompt_safety_gate=selected.task_prompt_safety_gate,
        task_contract_set=selected.task_contract_set,
        attachment_result=_attachment_result(selected),
        item_quality=selected.item_quality,
        batch_quality=batch,
        approval_policy=approval,
        decision_commits=(),
        relevant_revalidation_application_refs=(),
        revalidation_reports=(),
        current_head_refs=(),
    )


def _final_commit(source, candidate, decision: UserDecision):
    item_ref = evaluation_item_v2_ref(candidate.result.evaluation_item)
    request_ref = _ref("user-approval-request", decision.value.casefold())
    policy_ref = user_approval_policy_ref(source.approval_policy)
    record = UserDecisionRecord(
        decision_record_id="user-decision-record://pending",
        request_ref=request_ref,
        checkpoint=ApprovalCheckpoint.FINAL_DATASET_REVIEW,
        approval_policy_ref=policy_ref,
        subject_refs=(item_ref,),
        plan_ref=None,
        projection_ref=_ref("user-approval-projection", "final"),
        authenticated_user="requesting-user",
        decision=decision,
        reason="Final candidate decision.",
        hard_gate_override_requested=False,
        idempotency_key=f"final-{decision.value.casefold()}",
        decided_at=NOW,
        record_sha256="0" * 64,
    )
    digest = user_decision_record_carried_sha256(record)
    record = record.model_copy(
        update={
            "decision_record_id": f"user-decision-record://sha256/{digest}",
            "record_sha256": digest,
        }
    )
    outcome = {
        UserDecision.ACCEPT: UserDecisionCommitOutcomeV2.ACCEPTED,
        UserDecision.REJECT: UserDecisionCommitOutcomeV2.REJECTED,
        UserDecision.DEFER: UserDecisionCommitOutcomeV2.DEFERRED,
    }[decision]
    refs = (
        request_ref,
        _ref("user-approval-request-compilation-result", "final"),
        _ref("user-decision-handling-policy", "final"),
        _ref("authenticated-user-context", "requesting-user"),
    )
    return UserDecisionCommitResultV2.create(
        job_id=source.item.job_id,
        dataset_job_spec_ref=_ref("dataset-job-spec", "release-job"),
        request_compilation_result_ref=refs[1],
        request_ref=request_ref,
        decision_policy_ref=refs[2],
        authentication_context_ref=refs[3],
        decision_record=record,
        adjustment_effect=None,
        request_revision=None,
        outcome=outcome,
        audit=_audit(),
    )


def _complete_report(
    application_ref: ObjectRef,
    *,
    suffix: str,
    excluded_item_ids: tuple[str, ...] = (),
) -> DirectedRevalidationReportV2:
    work_result = RevalidationWorkResultV2.create(
        work_item_ref=_ref("revalidation-work-item", suffix),
        stage_result_ref=_ref("stage-result", suffix, version="record/v1"),
        outcome=RevalidationWorkOutcomeV2.SUCCEEDED,
        output_refs=(_ref("quality-report", f"replacement-{suffix}"),),
        failure_code=None,
        audit=_audit(),
    )
    return DirectedRevalidationReportV2.create(
        application_ref=application_ref,
        plan_ref=_ref("directed-revalidation-plan", suffix),
        work_results=(work_result,),
        replacement_current_refs=work_result.output_refs,
        invalidated_prior_refs=(_ref("quality-report", f"prior-{suffix}"),),
        excluded_item_ids=excluded_item_ids,
        required_checkpoint_reapprovals=(),
        outcome=DirectedRevalidationReportOutcomeV2.COMPLETE,
        audit=_audit(),
    )


@pytest.mark.asyncio
async def test_candidate_projects_query_and_complete_item_without_final_review() -> None:
    source = await _source(mode=ApprovalMode.NONE)
    compiler = ReleaseProjectionCompiler()

    candidate = compiler.compile_candidate(
        source=source,
        policy=_policy(),
        previous_result=None,
        audit=_audit(),
    )

    assert candidate.result.phase is ReleaseProjectionPhaseV2.CANDIDATE
    assert candidate.result.query_spec.prompt == source.task_draft.visible_prompt
    assert candidate.result.query_spec.attachment_dependency_ids == tuple(
        item.dependency_id for item in source.task_draft.attachment_dependencies
    )
    assert candidate.result.item_projection.item_status is ItemStatus.RUNNING
    terminal = compiler.finalize(
        candidate=candidate.result,
        source=source,
        policy=_policy(),
        audit=_audit(),
    )
    assert terminal.result.release_decision.state.value == "APPROVED"
    assert terminal.result.item_projection.item_status is ItemStatus.APPROVED
    assert terminal.result.evaluation_item.user_decision_record_refs == ()


@pytest.mark.asyncio
async def test_final_review_accept_and_reject_create_linked_terminal_decisions() -> None:
    source = await _source(
        mode=ApprovalMode.FINAL_ONLY,
        final_scope=FinalReviewScope.SELECTED_ITEMS,
    )
    compiler = ReleaseProjectionCompiler()
    candidate = compiler.compile_candidate(
        source=source,
        policy=_policy(),
        previous_result=None,
        audit=_audit(),
    )
    assert candidate.result.item_projection.item_status is ItemStatus.NEEDS_REVIEW

    accepted_commit = _final_commit(source, candidate, UserDecision.ACCEPT)
    accepted_source = replace(source, decision_commits=(accepted_commit,))
    approved = compiler.finalize(
        candidate=candidate.result,
        source=accepted_source,
        policy=_policy(),
        audit=_audit(),
    )
    assert approved.result.release_decision.state.value == "APPROVED"
    assert approved.result.release_decision.previous_decision_ref == (candidate.result.release_decision_ref)
    assert approved.result.release_subject_ref == candidate.result.release_subject_ref
    assert (
        approved.result.release_decision.release_subject_sha256
        == candidate.result.release_subject.release_subject_sha256
    )
    assert approved.result.evaluation_item.user_decision_record_refs == (accepted_commit.decision_record_ref,)

    rejected_commit = _final_commit(source, candidate, UserDecision.REJECT)
    rejected_source = replace(source, decision_commits=(rejected_commit,))
    rejected = compiler.finalize(
        candidate=candidate.result,
        source=rejected_source,
        policy=_policy(),
        audit=_audit(),
    )
    assert rejected.result.release_decision.state.value == "REJECTED"
    assert rejected.result.item_projection.item_status is ItemStatus.REJECTED
    assert rejected.result.release_subject_ref == candidate.result.release_subject_ref
    assert rejected.result.evaluation_item.user_decision_record_refs == (rejected_commit.decision_record_ref,)


@pytest.mark.asyncio
async def test_final_review_defer_remains_pending() -> None:
    source = await _source(
        mode=ApprovalMode.FINAL_ONLY,
        final_scope=FinalReviewScope.SELECTED_ITEMS,
    )
    compiler = ReleaseProjectionCompiler()
    candidate = compiler.compile_candidate(
        source=source,
        policy=_policy(),
        previous_result=None,
        audit=_audit(),
    )
    deferred = replace(
        source,
        decision_commits=(_final_commit(source, candidate, UserDecision.DEFER),),
    )
    with pytest.raises(ReleaseProjectionPendingError, match="pending"):
        compiler.finalize(
            candidate=candidate.result,
            source=deferred,
            policy=_policy(),
            audit=_audit(),
        )


@pytest.mark.asyncio
async def test_candidate_rejects_hard_gate_and_component_drift() -> None:
    source = await _source(mode=ApprovalMode.NONE)
    compiler = ReleaseProjectionCompiler()
    with pytest.raises(ReleaseProjectionPolicyError, match="stale or malformed"):
        compiler.compile_candidate(
            source=replace(
                source,
                batch_quality=source.batch_quality.model_copy(update={"approvable": False}),
            ),
            policy=_policy(),
            previous_result=None,
            audit=_audit(),
        )

    with pytest.raises(ReleaseProjectionPolicyError, match="R4 contract set"):
        compiler.compile_candidate(
            source=replace(
                source,
                task_contract_set=source.task_contract_set.model_copy(
                    update={"tool_policy_ref": _ref("tool-policy", "stale")}
                ),
            ),
            policy=_policy(),
            previous_result=None,
            audit=_audit(),
        )


@pytest.mark.asyncio
async def test_every_relevant_application_requires_current_complete_report() -> None:
    source = await _source(mode=ApprovalMode.NONE)
    first_app = _ref("user-plan-application", "first")
    second_app = _ref("user-plan-application", "second")
    first = _complete_report(first_app, suffix="first")
    second = _complete_report(second_app, suffix="second")
    complete = replace(
        source,
        relevant_revalidation_application_refs=(first_app, second_app),
        revalidation_reports=(first, second),
        current_head_refs=(
            *first.replacement_current_refs,
            *second.replacement_current_refs,
        ),
    )
    candidate = ReleaseProjectionCompiler().compile_candidate(
        source=complete,
        policy=_policy(),
        previous_result=None,
        audit=_audit(),
    )
    assert candidate.result.release_subject.revalidation_report_refs == tuple(
        sorted(
            (
                directed_revalidation_report_v2_ref(first),
                directed_revalidation_report_v2_ref(second),
            ),
            key=lambda value: value.object_id,
        )
    )

    with pytest.raises(ReleaseProjectionPolicyError, match="every relevant"):
        ReleaseProjectionCompiler().compile_candidate(
            source=replace(
                complete,
                revalidation_reports=(first,),
            ),
            policy=_policy(),
            previous_result=None,
            audit=_audit(),
        )

    excluded = _complete_report(
        second_app,
        suffix="excluded",
        excluded_item_ids=(source.item.item_id,),
    )
    excluded_reports = tuple(
        sorted(
            (first, excluded),
            key=lambda value: directed_revalidation_report_v2_ref(value).object_id,
        )
    )
    excluded_heads = tuple(
        sorted(
            {
                *first.replacement_current_refs,
                *excluded.replacement_current_refs,
            },
            key=lambda value: (
                value.object_type,
                value.object_id,
                value.object_version,
                value.object_sha256,
            ),
        )
    )
    with pytest.raises(ReleaseProjectionPolicyError, match="excluded"):
        ReleaseProjectionCompiler().compile_candidate(
            source=replace(
                complete,
                revalidation_reports=excluded_reports,
                current_head_refs=excluded_heads,
            ),
            policy=_policy(),
            previous_result=None,
            audit=_audit(),
        )


@pytest.mark.asyncio
async def test_release_chain_currentness_and_depth_fail_closed() -> None:
    source = await _source(mode=ApprovalMode.NONE)
    compiler = ReleaseProjectionCompiler()
    policy = _policy()
    candidate = compiler.compile_candidate(
        source=source,
        policy=policy,
        previous_result=None,
        audit=_audit(),
    ).result
    terminal = compiler.finalize(
        candidate=candidate,
        source=source,
        policy=policy,
        audit=_audit(),
    ).result

    compiler.validate_current(
        candidate,
        source=source,
        policy=policy,
        previous_result=None,
        audit=candidate.audit,
    )
    compiler.validate_current(
        terminal,
        source=source,
        policy=policy,
        previous_result=candidate,
        audit=terminal.audit,
    )
    with pytest.raises(
        ReleaseProjectionPolicyError,
        match="requires its candidate",
    ):
        compiler.validate_current(
            terminal,
            source=source,
            policy=policy,
            previous_result=None,
            audit=terminal.audit,
        )

    different_audit = _audit().model_copy(update={"created_by": "different-release-actor"})
    different = compiler.finalize(
        candidate=candidate,
        source=source,
        policy=policy,
        audit=different_audit,
    ).result
    with pytest.raises(
        ReleaseProjectionPolicyError,
        match="differs from current authority",
    ):
        compiler.validate_current(
            different,
            source=source,
            policy=policy,
            previous_result=candidate,
            audit=_audit(),
        )

    shallow = _policy_with(max_chain_depth=1)
    shallow_candidate = compiler.compile_candidate(
        source=source,
        policy=shallow,
        previous_result=None,
        audit=_audit(),
    ).result
    with pytest.raises(
        ReleaseProjectionPolicyError,
        match="chain depth",
    ):
        compiler.finalize(
            candidate=shallow_candidate,
            source=source,
            policy=shallow,
            audit=_audit(),
        )
    with pytest.raises(
        ReleaseProjectionPolicyError,
        match="chain depth",
    ):
        compiler.compile_candidate(
            source=source,
            policy=shallow,
            previous_result=shallow_candidate,
            audit=_audit(),
        )


@pytest.mark.asyncio
async def test_stale_policy_candidate_and_predecessor_are_rejected() -> None:
    source = await _source(mode=ApprovalMode.NONE)
    compiler = ReleaseProjectionCompiler()
    policy = _policy()
    candidate = compiler.compile_candidate(
        source=source,
        policy=policy,
        previous_result=None,
        audit=_audit(),
    ).result
    terminal = compiler.finalize(
        candidate=candidate,
        source=source,
        policy=policy,
        audit=_audit(),
    ).result

    stale_policy = policy.model_copy(update={"max_chain_depth": policy.max_chain_depth - 1})
    with pytest.raises(
        ReleaseProjectionPolicyError,
        match="policy is stale",
    ):
        compiler.compile_candidate(
            source=source,
            policy=stale_policy,
            previous_result=None,
            audit=_audit(),
        )
    with pytest.raises(
        ReleaseProjectionPolicyError,
        match="requires a candidate",
    ):
        compiler.finalize(
            candidate=terminal,
            source=source,
            policy=policy,
            audit=_audit(),
        )

    stale_candidate = candidate.model_copy(update={"result_sha256": "f" * 64})
    with pytest.raises(
        ReleaseProjectionPolicyError,
        match="candidate release projection is stale",
    ):
        compiler.finalize(
            candidate=stale_candidate,
            source=source,
            policy=policy,
            audit=_audit(),
        )
    with pytest.raises(
        ReleaseProjectionPolicyError,
        match="previous release projection is stale",
    ):
        compiler.compile_candidate(
            source=source,
            policy=policy,
            previous_result=stale_candidate,
            audit=_audit(),
        )

    alternate_policy = _policy_with(max_chain_depth=31)
    alternate_candidate = compiler.compile_candidate(
        source=source,
        policy=alternate_policy,
        previous_result=None,
        audit=_audit(),
    ).result
    with pytest.raises(
        ReleaseProjectionPolicyError,
        match="previous release projection policy is stale",
    ):
        compiler.compile_candidate(
            source=source,
            policy=policy,
            previous_result=alternate_candidate,
            audit=_audit(),
        )
    with pytest.raises(
        ReleaseProjectionPolicyError,
        match="candidate release projection policy is stale",
    ):
        compiler.finalize(
            candidate=alternate_candidate,
            source=source,
            policy=policy,
            audit=_audit(),
        )


@pytest.mark.asyncio
async def test_release_source_budget_and_status_matrix_fail_closed() -> None:
    source = await _source(mode=ApprovalMode.NONE)
    compiler = ReleaseProjectionCompiler()

    with pytest.raises(
        ReleaseProjectionPolicyError,
        match="running or awaiting review",
    ):
        compiler.compile_candidate(
            source=replace(
                source,
                item=source.item.model_copy(update={"status": ItemStatus.APPROVED}),
            ),
            policy=_policy(),
            previous_result=None,
            audit=_audit(),
        )
    with pytest.raises(
        ReleaseProjectionPolicyError,
        match="source trace inventory",
    ):
        compiler.compile_candidate(
            source=replace(source, source_trace_refs=()),
            policy=_policy(),
            previous_result=None,
            audit=_audit(),
        )
    with pytest.raises(
        ReleaseProjectionPolicyError,
        match="unique final matching",
    ):
        compiler.compile_candidate(
            source=replace(
                source,
                label_decisions=(
                    source.label_decisions[0],
                    source.label_decisions[0],
                ),
            ),
            policy=_policy(),
            previous_result=None,
            audit=_audit(),
        )
    with pytest.raises(
        ReleaseProjectionPolicyError,
        match="TaskDraft identity",
    ):
        compiler.compile_candidate(
            source=replace(
                source,
                task_draft=source.task_draft.model_copy(update={"visible_prompt": "stale prompt"}),
            ),
            policy=_policy(),
            previous_result=None,
            audit=_audit(),
        )
    with pytest.raises(
        ReleaseProjectionPolicyError,
        match="only enabled Environment Strategy",
    ):
        compiler.compile_candidate(
            source=replace(
                source,
                not_required_checkpoints=(ApprovalCheckpoint.LABEL_PLAN,),
            ),
            policy=_policy(),
            previous_result=None,
            audit=_audit(),
        )
    with pytest.raises(
        ReleaseProjectionPolicyError,
        match="current head ref budget",
    ):
        compiler.compile_candidate(
            source=replace(
                source,
                current_head_refs=(
                    _ref("quality-report", "head-a"),
                    _ref("quality-report", "head-b"),
                ),
            ),
            policy=_policy_with(max_current_head_refs=1),
            previous_result=None,
            audit=_audit(),
        )


@pytest.mark.asyncio
async def test_release_projection_gold_matches_closed_decision_paths() -> None:
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    cases = {case["scenario_id"]: case for case in payload["scenarios"]}
    compiler = ReleaseProjectionCompiler()

    source = await _source(mode=ApprovalMode.NONE)
    candidate = compiler.compile_candidate(
        source=source,
        policy=_policy(),
        previous_result=None,
        audit=_audit(),
    )
    terminal = compiler.finalize(
        candidate=candidate.result,
        source=source,
        policy=_policy(),
        audit=_audit(),
    )
    immediate = cases["final-disabled-immediate-approval"]
    assert candidate.result.release_decision.state.value == (immediate["candidate_state"])
    assert candidate.result.item_projection.item_status.value == (immediate["candidate_item_status"])
    assert terminal.result.release_decision.action.value == (immediate["expected_action"])
    assert terminal.result.release_decision.state.value == (immediate["expected_state"])

    final_source = await _source(
        mode=ApprovalMode.FINAL_ONLY,
        final_scope=FinalReviewScope.SELECTED_ITEMS,
    )
    final_candidate = compiler.compile_candidate(
        source=final_source,
        policy=_policy(),
        previous_result=None,
        audit=_audit(),
    )
    for decision, case_id in (
        (UserDecision.ACCEPT, "final-accepted"),
        (UserDecision.REJECT, "final-rejected"),
    ):
        commit = _final_commit(
            final_source,
            final_candidate,
            decision,
        )
        result = compiler.finalize(
            candidate=final_candidate.result,
            source=replace(
                final_source,
                decision_commits=(commit,),
            ),
            policy=_policy(),
            audit=_audit(),
        )
        expected = cases[case_id]
        assert result.result.release_decision.action.value == (expected["expected_action"])
        assert result.result.release_decision.state.value == (expected["expected_state"])

    deferred = _final_commit(
        final_source,
        final_candidate,
        UserDecision.DEFER,
    )
    with pytest.raises(ReleaseProjectionPendingError):
        compiler.finalize(
            candidate=final_candidate.result,
            source=replace(
                final_source,
                decision_commits=(deferred,),
            ),
            policy=_policy(),
            audit=_audit(),
        )

    serialized_keys: set[str] = set()

    def collect_keys(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                serialized_keys.add(str(key).casefold())
                collect_keys(child)
        elif isinstance(value, list):
            for child in value:
                collect_keys(child)

    collect_keys(payload)
    forbidden_markers = (
        "credential",
        "final_output",
        "grader",
        "hidden_condition",
        "private_reference",
        "prompt",
        "reason_text",
        "trace_content",
    )
    assert not any(marker in key for key in serialized_keys for marker in forbidden_markers)
