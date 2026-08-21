from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pytest
from test_job_store import _plan, _work_graph
from test_task_rewrite import _compile_plan, _source_chain
from test_user_approval_requests import (
    _compile,
    _environment_source,
    _final_source,
    _generation_policy,
    _job_spec,
    _label_source,
    _policy,
)
from test_user_decisions import (
    _audit,
    _authentication,
    _handling_policy,
    _label_request,
)
from test_user_plan_revalidation import (
    _application_policy,
    _ref,
)

from eval_factory.approval.adjustments import LabelPlanAdjustmentCompiler
from eval_factory.approval.application_persistence import (
    UserPlanApplicationPersistenceService,
)
from eval_factory.approval.interaction import (
    UserCheckpointAuthenticationError,
    UserCheckpointInteractionConflictError,
    UserCheckpointInteractionError,
    UserCheckpointInteractionNotResumableError,
    UserCheckpointInteractionService,
)
from eval_factory.approval.interaction_models import (
    UserCheckpointSourceContextV2,
)
from eval_factory.approval.interaction_store import (
    UserCheckpointMaterialStore,
)
from eval_factory.approval.persistence import (
    UserDecisionPersistenceService,
)
from eval_factory.approval.requests import (
    LabelPlanApprovalSource,
    TaskRewriteApprovalSource,
)
from eval_factory.approval.revalidation import (
    RevalidationItemSource,
    RevalidationStageSource,
)
from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    ApprovalMode,
    FinalReviewScope,
    TypedAdjustment,
    UserDecision,
)
from eval_factory.contracts.approval_v2 import (
    label_plan_carried_sha256,
)
from eval_factory.contracts.checkpoint_interaction_v2 import (
    UserCheckpointDecisionResultV2,
    UserCheckpointDecisionSubmissionV2,
    UserCheckpointInteractionPolicyV2,
    UserCheckpointInteractionStateV2,
    UserCheckpointInteractionV2,
    UserCheckpointOpenOutcomeV2,
    UserCheckpointOpenResultV2,
    UserCheckpointResumeDispositionV2,
    UserCheckpointResumeResultV2,
    user_checkpoint_decision_result_v2_ref,
    user_checkpoint_interaction_v2_ref,
    user_checkpoint_open_result_v2_ref,
    user_checkpoint_resume_result_v2_ref,
)
from eval_factory.contracts.cli_v2 import (
    PipelineControlConfigV2,
    PipelineResumeActionV2,
)
from eval_factory.contracts.orchestration import JobStatus
from eval_factory.contracts.orchestration_v2 import StageNameV2
from eval_factory.orchestration.job_store import (
    ConcurrencyConflictError,
    IdempotencyConflictError,
    ImmutableResultError,
    JobStore,
    RecordNotFoundError,
)
from eval_factory.orchestration.lifecycle import (
    PipelineCheckpointRequiredError,
    PipelineLifecycleService,
)


def _interaction_policy() -> UserCheckpointInteractionPolicyV2:
    return UserCheckpointInteractionPolicyV2.create(
        max_interactions_per_job=100,
        max_requests_per_interaction=10,
        max_page_size=100,
        max_presentation_bytes=1_000_000,
        max_source_context_bytes=2_000_000,
        max_ready_work_refs=100,
        audit=_audit(),
    )


def _service(
    tmp_path: Path,
    *,
    policy,
) -> tuple[JobStore, UserCheckpointInteractionService]:
    store = JobStore(tmp_path / "factory.sqlite3")
    store.create_job(_job_spec(policy))
    material = UserCheckpointMaterialStore(
        tmp_path / "checkpoint-store",
        max_source_context_bytes=2_000_000,
        max_presentation_bytes=1_000_000,
    )
    return store, UserCheckpointInteractionService(store, material)


def _row_count(store: JobStore, table: str) -> int:
    with sqlite3.connect(store.path) as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _opened_label(
    tmp_path: Path,
    *,
    sources=None,
    key: str = "open-r7-07-helper",
):
    approval_policy = _policy(ApprovalMode.PLAN_GATES)
    source_values = sources or (_label_source(),)
    compilation = _compile(
        policy=approval_policy,
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        sources=source_values,
    )
    store, service = _service(tmp_path, policy=approval_policy)
    running = store.transition_job(
        compilation.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key=f"start-{key}",
    )
    context = UserCheckpointSourceContextV2.create(
        job_spec=_job_spec(approval_policy),
        approval_policy=approval_policy,
        generation_policy=_generation_policy(),
        handling_policy=_handling_policy(),
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=source_values,
    )
    opened = service.open_checkpoint(
        source_context=context,
        policy=_interaction_policy(),
        expected_job_version=running.row_version,
        idempotency_key=key,
        audit=_audit(),
    )
    assert opened.interaction is not None
    return store, service, context, opened


def _successor_label_source(
    context: UserCheckpointSourceContextV2,
    *,
    boundary: str,
) -> LabelPlanApprovalSource:
    plan = context.label_plans[0].model_copy(
        update={
            "label_plan_id": "label-plan://pending",
            "boundary": boundary,
        }
    )
    digest = label_plan_carried_sha256(plan)
    return LabelPlanApprovalSource(
        label_spec=context.label_specs[0],
        label_plan=plan.model_copy(update={"label_plan_id": f"label-plan://sha256/{digest}"}),
    )


def _successor_label_context(
    context: UserCheckpointSourceContextV2,
    *,
    boundary: str,
) -> UserCheckpointSourceContextV2:
    source = _successor_label_source(context, boundary=boundary)
    compilation = _compile(
        policy=context.approval_policy,
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        sources=(source,),
    )
    return UserCheckpointSourceContextV2.create(
        job_spec=context.job_spec,
        approval_policy=context.approval_policy,
        generation_policy=context.generation_policy,
        handling_policy=context.handling_policy,
        request_compilation=compilation,
        requested_by=context.requested_by,
        sources=(source,),
    )


def _interaction_with_audit(
    interaction: UserCheckpointInteractionV2,
    *,
    created_by: str,
) -> UserCheckpointInteractionV2:
    audit = interaction.audit.model_copy(
        update={
            "created_at": interaction.audit.created_at + timedelta(days=1),
            "created_by": created_by,
        }
    )
    return UserCheckpointInteractionV2.create(
        job_id=interaction.job_id,
        chain_id=interaction.chain_id,
        interaction_version=interaction.interaction_version,
        predecessor_interaction_ref=interaction.predecessor_interaction_ref,
        checkpoint=interaction.checkpoint,
        request_compilation_ref=interaction.request_compilation_ref,
        request_ref=interaction.request_ref,
        presentation_ref=interaction.presentation_ref,
        source_context_ref=interaction.source_context_ref,
        state=interaction.state,
        decision_commit_ref=interaction.decision_commit_ref,
        application_ref=interaction.application_ref,
        job_status=interaction.job_status,
        occurred_at=interaction.occurred_at,
        policy_ref=interaction.policy_ref,
        audit=audit,
    )


def test_open_requested_checkpoint_atomically_blocks_job(
    tmp_path: Path,
) -> None:
    approval_policy, source, compilation, _ = _label_request()
    store, service = _service(tmp_path, policy=approval_policy)
    running = store.transition_job(
        compilation.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-r7-07-open",
    )
    context = UserCheckpointSourceContextV2.create(
        job_spec=_job_spec(approval_policy),
        approval_policy=approval_policy,
        generation_policy=_generation_policy(),
        handling_policy=_handling_policy(),
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=(source,),
    )

    result = service.open_checkpoint(
        source_context=context,
        policy=_interaction_policy(),
        expected_job_version=running.row_version,
        idempotency_key="open-r7-07-label",
        audit=_audit(),
    )

    assert result.outcome is UserCheckpointOpenOutcomeV2.PAUSED
    assert result.interaction is not None
    assert result.interaction.state is (UserCheckpointInteractionStateV2.PENDING_DECISION)
    assert store.get_job(compilation.job_id).status is JobStatus.BLOCKED
    assert service.get_current_interaction(compilation.job_id) == (result.interaction)
    assert (
        service.show_interaction(result.interaction.interaction_id).presentation == context.presentations()[0]
    )


def test_every_checkpoint_opens_its_exact_safe_presentation(
    tmp_path: Path,
) -> None:
    plan_policy = _policy(ApprovalMode.PLAN_GATES)
    rewrite_compilation = _compile_plan(_source_chain())[1]
    assert rewrite_compilation.plan_version is not None
    assert rewrite_compilation.preview_safety_gate is not None
    assert rewrite_compilation.preview is not None
    rewrite_source = TaskRewriteApprovalSource(
        plan_version=rewrite_compilation.plan_version,
        preview_safety_gate=rewrite_compilation.preview_safety_gate,
        preview=rewrite_compilation.preview,
    )
    final_policy = _policy(
        ApprovalMode.FINAL_ONLY,
        scope=FinalReviewScope.PROMPTS,
    )
    cases = (
        (
            ApprovalCheckpoint.LABEL_PLAN,
            plan_policy,
            (_label_source(),),
            "label_plan",
        ),
        (
            ApprovalCheckpoint.TASK_REWRITE_PLAN,
            plan_policy,
            (rewrite_source,),
            "task_rewrite_preview",
        ),
        (
            ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
            plan_policy,
            (_environment_source(),),
            "environment_strategy",
        ),
        (
            ApprovalCheckpoint.FINAL_DATASET_REVIEW,
            final_policy,
            (_final_source(FinalReviewScope.PROMPTS),),
            "final_dataset_review_preview",
        ),
    )
    for checkpoint, policy, sources, body_field in cases:
        case_root = tmp_path / checkpoint.value.casefold()
        case_root.mkdir()
        compilation = _compile(
            policy=policy,
            checkpoint=checkpoint,
            sources=sources,
        )
        store, service = _service(case_root, policy=policy)
        running = store.transition_job(
            compilation.job_id,
            JobStatus.RUNNING,
            expected_version=0,
            idempotency_key=f"start-r7-07-{checkpoint.value.casefold()}",
        )
        context = UserCheckpointSourceContextV2.create(
            job_spec=_job_spec(policy),
            approval_policy=policy,
            generation_policy=_generation_policy(),
            handling_policy=_handling_policy(),
            request_compilation=compilation,
            requested_by="requesting-user",
            sources=sources,
        )

        opened = service.open_checkpoint(
            source_context=context,
            policy=_interaction_policy(),
            expected_job_version=running.row_version,
            idempotency_key=f"open-r7-07-{checkpoint.value.casefold()}",
            audit=_audit(),
        )

        assert opened.interaction is not None
        shown = service.show_interaction(opened.interaction.interaction_id)
        assert getattr(shown.presentation, body_field) is not None
        bodies = (
            shown.presentation.label_plan,
            shown.presentation.task_rewrite_preview,
            shown.presentation.environment_strategy,
            shown.presentation.final_dataset_review_preview,
        )
        assert sum(value is not None for value in bodies) == 1


def test_environment_not_required_creates_no_material_or_pause(
    tmp_path: Path,
) -> None:
    approval_policy = _policy(ApprovalMode.PLAN_GATES)
    compilation = _compile(
        policy=approval_policy,
        checkpoint=ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        sources=(),
    )
    store, service = _service(tmp_path, policy=approval_policy)
    running = store.transition_job(
        compilation.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-r7-07-not-required",
    )
    context = UserCheckpointSourceContextV2.create(
        job_spec=_job_spec(approval_policy),
        approval_policy=approval_policy,
        generation_policy=_generation_policy(),
        handling_policy=_handling_policy(),
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=(),
    )

    result = service.open_checkpoint(
        source_context=context,
        policy=_interaction_policy(),
        expected_job_version=running.row_version,
        idempotency_key="open-r7-07-not-required",
        audit=_audit(),
    )

    assert result.outcome is UserCheckpointOpenOutcomeV2.NOT_REQUIRED
    assert result.interaction is None
    assert store.get_job(compilation.job_id).status is JobStatus.RUNNING
    assert (
        service.list_interactions(
            compilation.job_id,
            offset=0,
            limit=100,
        ).total
        == 0
    )
    assert not (tmp_path / "checkpoint-store" / "source-context").exists()
    assert not (tmp_path / "checkpoint-store" / "presentation").exists()


def test_disabled_checkpoint_creates_no_pause_or_interaction(
    tmp_path: Path,
) -> None:
    approval_policy = _policy(ApprovalMode.NONE)
    compilation = _compile(
        policy=approval_policy,
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        sources=(),
    )
    store, service = _service(tmp_path, policy=approval_policy)
    running = store.transition_job(
        compilation.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-r7-07-disabled",
    )
    context = UserCheckpointSourceContextV2.create(
        job_spec=_job_spec(approval_policy),
        approval_policy=approval_policy,
        generation_policy=_generation_policy(),
        handling_policy=_handling_policy(),
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=(),
    )

    result = service.open_checkpoint(
        source_context=context,
        policy=_interaction_policy(),
        expected_job_version=running.row_version,
        idempotency_key="open-r7-07-disabled",
        audit=_audit(),
    )

    assert result.outcome is UserCheckpointOpenOutcomeV2.DISABLED
    assert result.interaction is None
    assert store.get_job(compilation.job_id).status is JobStatus.RUNNING
    assert service.list_interactions(compilation.job_id, offset=0, limit=100).total == 0


def test_accept_is_committed_then_explicitly_resumed(
    tmp_path: Path,
) -> None:
    approval_policy, source, compilation, _ = _label_request()
    store, service = _service(tmp_path, policy=approval_policy)
    running = store.transition_job(
        compilation.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-r7-07-accept",
    )
    context = UserCheckpointSourceContextV2.create(
        job_spec=_job_spec(approval_policy),
        approval_policy=approval_policy,
        generation_policy=_generation_policy(),
        handling_policy=_handling_policy(),
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=(source,),
    )
    opened = service.open_checkpoint(
        source_context=context,
        policy=_interaction_policy(),
        expected_job_version=running.row_version,
        idempotency_key="open-r7-07-accept",
        audit=_audit(),
    )
    assert opened.interaction is not None
    submission = UserCheckpointDecisionSubmissionV2(
        interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
        decision=UserDecision.ACCEPT,
        reason="Accept the current checkpoint.",
        idempotency_key="decide-r7-07-accept",
    )

    decided = service.decide(
        interaction_id=opened.interaction.interaction_id,
        submission=submission,
        authentication=_authentication(),
        expected_job_version=opened.job_version,
        audit=_audit(),
    )

    assert decided.interaction.state is (UserCheckpointInteractionStateV2.RESUMABLE)
    assert decided.resume_disposition is (UserCheckpointResumeDispositionV2.DIRECT)
    assert store.get_job(compilation.job_id).status is JobStatus.BLOCKED
    resumed = service.resume_checkpoint(
        job_id=compilation.job_id,
        expected_job_version=decided.job_version,
        idempotency_key="resume-r7-07-accept",
        audit=_audit(),
    )
    assert resumed.interaction.state is (UserCheckpointInteractionStateV2.RESUMED)
    assert resumed.job_status is JobStatus.RUNNING
    assert store.list_stage_runs(job_id=compilation.job_id) == ()
    assert store.list_work_leases(job_id=compilation.job_id) == ()


def test_plan_reject_terminates_and_generic_resume_cannot_bypass_pending(
    tmp_path: Path,
) -> None:
    approval_policy, source, compilation, _ = _label_request()
    store, service = _service(tmp_path, policy=approval_policy)
    running = store.transition_job(
        compilation.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-r7-07-reject",
    )
    context = UserCheckpointSourceContextV2.create(
        job_spec=_job_spec(approval_policy),
        approval_policy=approval_policy,
        generation_policy=_generation_policy(),
        handling_policy=_handling_policy(),
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=(source,),
    )
    opened = service.open_checkpoint(
        source_context=context,
        policy=_interaction_policy(),
        expected_job_version=running.row_version,
        idempotency_key="open-r7-07-reject",
        audit=_audit(),
    )
    assert opened.interaction is not None
    with pytest.raises(PipelineCheckpointRequiredError):
        PipelineLifecycleService(store).resume(
            job_id=compilation.job_id,
            expected_job_version=opened.job_version,
            idempotency_key="bypass-r7-07-reject",
            offset=0,
            limit=100,
        )
    rejected = service.decide(
        interaction_id=opened.interaction.interaction_id,
        submission=UserCheckpointDecisionSubmissionV2(
            interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
            decision=UserDecision.REJECT,
            reason="Reject the current label plan.",
            idempotency_key="decide-r7-07-reject",
        ),
        authentication=_authentication(),
        expected_job_version=opened.job_version,
        audit=_audit(),
    )

    assert rejected.interaction.state is (UserCheckpointInteractionStateV2.TERMINATED)
    assert rejected.resume_disposition is (UserCheckpointResumeDispositionV2.TERMINATED)
    assert store.get_job(compilation.job_id).status is JobStatus.FAILED


def test_open_decision_and_resume_exact_replay(
    tmp_path: Path,
) -> None:
    store, service, context, opened = _opened_label(
        tmp_path,
        key="open-r7-07-replay",
    )
    replay = service.open_checkpoint(
        source_context=context,
        policy=_interaction_policy(),
        expected_job_version=1,
        idempotency_key="open-r7-07-replay",
        audit=_audit(),
    )
    assert replay == opened
    assert (
        service.list_interactions(
            context.job_spec.job_id,
            offset=0,
            limit=100,
        ).total
        == 1
    )
    with pytest.raises(IdempotencyConflictError):
        service.open_checkpoint(
            source_context=context,
            policy=_interaction_policy(),
            expected_job_version=2,
            idempotency_key="open-r7-07-replay",
            audit=_audit(),
        )
    assert opened.interaction is not None
    submission = UserCheckpointDecisionSubmissionV2(
        interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
        decision=UserDecision.ACCEPT,
        reason="Accept replay.",
        idempotency_key="decide-r7-07-replay",
    )
    decided = service.decide(
        interaction_id=opened.interaction.interaction_id,
        submission=submission,
        authentication=_authentication(),
        expected_job_version=opened.job_version,
        audit=_audit(),
    )
    assert (
        service.decide(
            interaction_id=opened.interaction.interaction_id,
            submission=submission,
            authentication=_authentication(),
            expected_job_version=opened.job_version,
            audit=_audit(),
        )
        == decided
    )
    resumed = service.resume_checkpoint(
        job_id=context.job_spec.job_id,
        expected_job_version=decided.job_version,
        idempotency_key="resume-r7-07-replay",
        audit=_audit(),
    )
    assert (
        service.resume_checkpoint(
            job_id=context.job_spec.job_id,
            expected_job_version=decided.job_version,
            idempotency_key="resume-r7-07-replay",
            audit=_audit(),
        )
        == resumed
    )
    assert service.get_result(opened.result_id) == opened
    assert service.get_result(decided.result_id) == decided
    assert service.get_result(resumed.result_id) == resumed
    assert store.get_job(context.job_spec.job_id).status is JobStatus.RUNNING


def test_open_fault_rolls_back_interaction_job_outbox_and_idempotency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approval_policy, source, compilation, _ = _label_request()
    store, service = _service(tmp_path, policy=approval_policy)
    running = store.transition_job(
        compilation.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-r7-07-open-fault",
    )
    context = UserCheckpointSourceContextV2.create(
        job_spec=_job_spec(approval_policy),
        approval_policy=approval_policy,
        generation_policy=_generation_policy(),
        handling_policy=_handling_policy(),
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=(source,),
    )

    def fail_result(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected checkpoint open result fault")

    monkeypatch.setattr(service, "_persist_result", fail_result)
    with pytest.raises(RuntimeError, match="injected"):
        service.open_checkpoint(
            source_context=context,
            policy=_interaction_policy(),
            expected_job_version=running.row_version,
            idempotency_key="open-r7-07-fault",
            audit=_audit(),
        )

    assert store.get_job(compilation.job_id) == running
    assert _row_count(store, "user_checkpoint_interactions") == 0
    assert _row_count(store, "user_checkpoint_results") == 0
    assert _row_count(store, "user_approval_requests") == 0
    assert all(event.event_type != "user-checkpoint-opened" for event in store.list_outbox())
    assert (tmp_path / "checkpoint-store" / "source-context").exists()


def test_decision_fault_rolls_back_r7_05_and_interaction_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, service, context, opened = _opened_label(
        tmp_path,
        key="open-r7-07-decision-fault",
    )
    assert opened.interaction is not None

    def fail_result(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected checkpoint decision result fault")

    monkeypatch.setattr(service, "_persist_decision_result", fail_result)
    with pytest.raises(RuntimeError, match="injected"):
        service.decide(
            interaction_id=opened.interaction.interaction_id,
            submission=UserCheckpointDecisionSubmissionV2(
                interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
                decision=UserDecision.ACCEPT,
                reason="Accept before the injected decision fault.",
                idempotency_key="decide-r7-07-fault",
            ),
            authentication=_authentication(),
            expected_job_version=opened.job_version,
            audit=_audit(),
        )

    assert store.get_job(context.job_spec.job_id).status is JobStatus.BLOCKED
    assert service.get_current_interaction(context.job_spec.job_id) == (opened.interaction)
    assert _row_count(store, "user_decision_records") == 0
    assert _row_count(store, "user_decision_commits") == 0
    assert _row_count(store, "user_checkpoint_interactions") == 1
    assert all(event.event_type != "user-decision-committed" for event in store.list_outbox())


def test_resume_fault_rolls_back_job_interaction_head_and_outbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, service, context, opened = _opened_label(
        tmp_path,
        key="open-r7-07-resume-fault",
    )
    assert opened.interaction is not None
    decided = service.decide(
        interaction_id=opened.interaction.interaction_id,
        submission=UserCheckpointDecisionSubmissionV2(
            interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
            decision=UserDecision.ACCEPT,
            reason="Accept before the injected resume fault.",
            idempotency_key="decide-r7-07-resume-fault",
        ),
        authentication=_authentication(),
        expected_job_version=opened.job_version,
        audit=_audit(),
    )

    def fail_result(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected checkpoint resume result fault")

    monkeypatch.setattr(service, "_persist_resume_result", fail_result)
    with pytest.raises(RuntimeError, match="injected"):
        service.resume_checkpoint(
            job_id=context.job_spec.job_id,
            expected_job_version=decided.job_version,
            idempotency_key="resume-r7-07-fault",
            audit=_audit(),
        )

    assert store.get_job(context.job_spec.job_id).status is JobStatus.BLOCKED
    assert service.get_current_interaction(context.job_spec.job_id) == (decided.interaction)
    assert _row_count(store, "user_checkpoint_interactions") == 2
    assert _row_count(store, "user_checkpoint_results") == 2
    assert all(event.event_type != "user-checkpoint-resumed" for event in store.list_outbox())


def test_two_checkpoint_open_workers_commit_one_authority(
    tmp_path: Path,
) -> None:
    approval_policy, source, compilation, _ = _label_request()
    store, service = _service(tmp_path, policy=approval_policy)
    running = store.transition_job(
        compilation.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-r7-07-open-race",
    )
    context = UserCheckpointSourceContextV2.create(
        job_spec=_job_spec(approval_policy),
        approval_policy=approval_policy,
        generation_policy=_generation_policy(),
        handling_policy=_handling_policy(),
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=(source,),
    )

    def attempt(key: str) -> UserCheckpointOpenResultV2 | Exception:
        try:
            return service.open_checkpoint(
                source_context=context,
                policy=_interaction_policy(),
                expected_job_version=running.row_version,
                idempotency_key=key,
                audit=_audit(),
            )
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(
                attempt,
                ("open-r7-07-race-a", "open-r7-07-race-b"),
            )
        )

    successes = tuple(value for value in results if isinstance(value, UserCheckpointOpenResultV2))
    failures = tuple(value for value in results if isinstance(value, Exception))
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(
        failures[0],
        (
            ConcurrencyConflictError,
            UserCheckpointInteractionConflictError,
        ),
    )
    assert _row_count(store, "user_checkpoint_interactions") == 1
    assert store.get_job(compilation.job_id).status is JobStatus.BLOCKED
    assert [event.event_type for event in store.list_outbox()].count("user-checkpoint-opened") == 1


def test_two_checkpoint_decision_workers_commit_one_mapping(
    tmp_path: Path,
) -> None:
    store, service, context, opened = _opened_label(
        tmp_path,
        key="open-r7-07-decision-race",
    )
    assert opened.interaction is not None

    def attempt(
        value: tuple[UserDecision, str],
    ) -> UserCheckpointDecisionResultV2 | Exception:
        decision, key = value
        try:
            return service.decide(
                interaction_id=opened.interaction.interaction_id,
                submission=UserCheckpointDecisionSubmissionV2(
                    interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
                    decision=decision,
                    reason=f"Concurrent {decision.value} decision.",
                    idempotency_key=key,
                ),
                authentication=_authentication(),
                expected_job_version=opened.job_version,
                audit=_audit(),
            )
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(
                attempt,
                (
                    (UserDecision.ACCEPT, "decide-r7-07-race-accept"),
                    (UserDecision.REJECT, "decide-r7-07-race-reject"),
                ),
            )
        )

    successes = tuple(value for value in results if isinstance(value, UserCheckpointDecisionResultV2))
    failures = tuple(value for value in results if isinstance(value, Exception))
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], UserCheckpointInteractionConflictError)
    assert _row_count(store, "user_decision_records") == 1
    assert _row_count(store, "user_decision_commits") == 1
    assert _row_count(store, "user_checkpoint_interactions") == 2
    assert [event.event_type for event in store.list_outbox()].count("user-decision-committed") == 1
    assert [event.event_type for event in store.list_outbox()].count("user-checkpoint-decision-mapped") == 1
    assert service.get_current_interaction(context.job_spec.job_id).state in {
        UserCheckpointInteractionStateV2.RESUMABLE,
        UserCheckpointInteractionStateV2.TERMINATED,
    }


def test_two_checkpoint_resume_workers_commit_one_transition(
    tmp_path: Path,
) -> None:
    store, service, context, opened = _opened_label(
        tmp_path,
        key="open-r7-07-resume-race",
    )
    assert opened.interaction is not None
    decided = service.decide(
        interaction_id=opened.interaction.interaction_id,
        submission=UserCheckpointDecisionSubmissionV2(
            interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
            decision=UserDecision.ACCEPT,
            reason="Accept before concurrent resume.",
            idempotency_key="decide-r7-07-resume-race",
        ),
        authentication=_authentication(),
        expected_job_version=opened.job_version,
        audit=_audit(),
    )

    def attempt(key: str) -> UserCheckpointResumeResultV2 | Exception:
        try:
            return service.resume_checkpoint(
                job_id=context.job_spec.job_id,
                expected_job_version=decided.job_version,
                idempotency_key=key,
                audit=_audit(),
            )
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(
                attempt,
                ("resume-r7-07-race-a", "resume-r7-07-race-b"),
            )
        )

    successes = tuple(value for value in results if isinstance(value, UserCheckpointResumeResultV2))
    failures = tuple(value for value in results if isinstance(value, Exception))
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(
        failures[0],
        (
            UserCheckpointInteractionConflictError,
            UserCheckpointInteractionNotResumableError,
        ),
    )
    assert store.get_job(context.job_spec.job_id).status is JobStatus.RUNNING
    assert (
        service.get_current_interaction(context.job_spec.job_id).state
        is UserCheckpointInteractionStateV2.RESUMED
    )
    assert [event.event_type for event in store.list_outbox()].count("user-checkpoint-resumed") == 1


def test_open_decision_and_resume_result_identity_uses_nested_refs(
    tmp_path: Path,
) -> None:
    _, service, context, opened = _opened_label(
        tmp_path,
        key="open-r7-07-audit-independent-results",
    )
    assert opened.interaction is not None
    decided = service.decide(
        interaction_id=opened.interaction.interaction_id,
        submission=UserCheckpointDecisionSubmissionV2(
            interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
            decision=UserDecision.ACCEPT,
            reason="Accept for audit identity coverage.",
            idempotency_key="decide-r7-07-audit-independent-results",
        ),
        authentication=_authentication(),
        expected_job_version=opened.job_version,
        audit=_audit(),
    )
    resumed = service.resume_checkpoint(
        job_id=context.job_spec.job_id,
        expected_job_version=decided.job_version,
        idempotency_key="resume-r7-07-audit-independent-results",
        audit=_audit(),
    )

    changed_open_interaction = _interaction_with_audit(
        opened.interaction,
        created_by="alternate-open-actor",
    )
    changed_open = UserCheckpointOpenResultV2.create(
        job_id=opened.job_id,
        outcome=opened.outcome,
        interaction=changed_open_interaction,
        job_status=opened.job_status,
        job_version=opened.job_version,
        policy_ref=opened.policy_ref,
        audit=opened.audit.model_copy(
            update={
                "created_at": opened.audit.created_at + timedelta(days=1),
                "created_by": "alternate-open-result-actor",
            }
        ),
    )
    changed_decision_interaction = _interaction_with_audit(
        decided.interaction,
        created_by="alternate-decision-actor",
    )
    changed_decision = UserCheckpointDecisionResultV2.create(
        predecessor_interaction_ref=decided.predecessor_interaction_ref,
        interaction=changed_decision_interaction,
        next_interaction=None,
        decision_commit=decided.decision_commit,
        resume_disposition=decided.resume_disposition,
        job_status=decided.job_status,
        job_version=decided.job_version,
        audit=decided.audit.model_copy(
            update={
                "created_at": decided.audit.created_at + timedelta(days=1),
                "created_by": "alternate-decision-result-actor",
            }
        ),
    )
    changed_resume_interaction = _interaction_with_audit(
        resumed.interaction,
        created_by="alternate-resume-actor",
    )
    changed_resume = UserCheckpointResumeResultV2.create(
        predecessor_interaction_ref=resumed.predecessor_interaction_ref,
        interaction=changed_resume_interaction,
        decision_commit_ref=resumed.decision_commit_ref,
        application_ref=resumed.application_ref,
        revalidation_report_ref=resumed.revalidation_report_ref,
        incomplete_work_refs=resumed.incomplete_work_refs,
        revalidation_ready_work_refs=resumed.revalidation_ready_work_refs,
        job_version=resumed.job_version,
        audit=resumed.audit.model_copy(
            update={
                "created_at": resumed.audit.created_at + timedelta(days=1),
                "created_by": "alternate-resume-result-actor",
            }
        ),
    )

    assert user_checkpoint_interaction_v2_ref(changed_open_interaction) == (
        user_checkpoint_interaction_v2_ref(opened.interaction)
    )
    assert user_checkpoint_open_result_v2_ref(changed_open) == (user_checkpoint_open_result_v2_ref(opened))
    assert user_checkpoint_decision_result_v2_ref(changed_decision) == (
        user_checkpoint_decision_result_v2_ref(decided)
    )
    assert user_checkpoint_resume_result_v2_ref(changed_resume) == (
        user_checkpoint_resume_result_v2_ref(resumed)
    )


def test_multiple_requests_advance_one_pending_head_at_a_time(
    tmp_path: Path,
) -> None:
    sources = (
        _label_source("first"),
        _label_source("second"),
    )
    _, service, context, opened = _opened_label(
        tmp_path,
        sources=sources,
        key="open-r7-07-multiple",
    )
    assert opened.interaction is not None
    first = service.decide(
        interaction_id=opened.interaction.interaction_id,
        submission=UserCheckpointDecisionSubmissionV2(
            interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
            decision=UserDecision.ACCEPT,
            reason="Accept first request.",
            idempotency_key="decide-r7-07-first",
        ),
        authentication=_authentication(),
        expected_job_version=opened.job_version,
        audit=_audit(),
    )
    assert first.next_interaction is not None
    assert first.resume_disposition is (UserCheckpointResumeDispositionV2.NEXT_REQUEST)
    assert service.get_current_interaction(context.job_spec.job_id) == (first.next_interaction)
    second = service.decide(
        interaction_id=first.next_interaction.interaction_id,
        submission=UserCheckpointDecisionSubmissionV2(
            interaction_ref=user_checkpoint_interaction_v2_ref(first.next_interaction),
            decision=UserDecision.ACCEPT,
            reason="Accept second request.",
            idempotency_key="decide-r7-07-second",
        ),
        authentication=_authentication(),
        expected_job_version=first.job_version,
        audit=_audit(),
    )
    assert second.next_interaction is None
    assert second.interaction.state is (UserCheckpointInteractionStateV2.RESUMABLE)
    misbound_interaction = UserCheckpointInteractionV2.create(
        job_id=first.interaction.job_id,
        chain_id=first.interaction.chain_id,
        interaction_version=first.interaction.interaction_version,
        predecessor_interaction_ref=first.interaction.predecessor_interaction_ref,
        checkpoint=first.interaction.checkpoint,
        request_compilation_ref=first.interaction.request_compilation_ref,
        request_ref=first.interaction.request_ref,
        presentation_ref=first.interaction.presentation_ref,
        source_context_ref=first.interaction.source_context_ref,
        state=first.interaction.state,
        decision_commit_ref=second.interaction.decision_commit_ref,
        application_ref=None,
        job_status=first.interaction.job_status,
        occurred_at=first.interaction.occurred_at,
        policy_ref=first.interaction.policy_ref,
        audit=_audit(),
    )
    with pytest.raises(ValueError, match="decision result interaction binding"):
        UserCheckpointDecisionResultV2.create(
            predecessor_interaction_ref=first.predecessor_interaction_ref,
            interaction=misbound_interaction,
            next_interaction=None,
            decision_commit=second.decision_commit,
            resume_disposition=UserCheckpointResumeDispositionV2.DIRECT,
            job_status=JobStatus.BLOCKED,
            job_version=second.job_version,
            audit=_audit(),
        )
    assert (
        service.list_interactions(
            context.job_spec.job_id,
            offset=0,
            limit=100,
        ).total
        == 4
    )


def test_resumed_checkpoint_can_open_a_later_current_request(
    tmp_path: Path,
) -> None:
    store, service, context, opened = _opened_label(
        tmp_path,
        key="open-r7-07-later",
    )
    assert opened.interaction is not None
    decided = service.decide(
        interaction_id=opened.interaction.interaction_id,
        submission=UserCheckpointDecisionSubmissionV2(
            interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
            decision=UserDecision.ACCEPT,
            reason="Accept the first checkpoint.",
            idempotency_key="decide-r7-07-later",
        ),
        authentication=_authentication(),
        expected_job_version=opened.job_version,
        audit=_audit(),
    )
    resumed = service.resume_checkpoint(
        job_id=context.job_spec.job_id,
        expected_job_version=decided.job_version,
        idempotency_key="resume-r7-07-later",
        audit=_audit(),
    )
    successor_context = _successor_label_context(
        context,
        boundary="Use the revalidated current tool facts only.",
    )

    later = service.open_checkpoint(
        source_context=successor_context,
        policy=_interaction_policy(),
        expected_job_version=resumed.job_version,
        idempotency_key="open-r7-07-later-successor",
        audit=_audit(),
    )

    assert later.interaction is not None
    assert later.interaction.predecessor_interaction_ref == (
        user_checkpoint_interaction_v2_ref(resumed.interaction)
    )
    assert later.interaction.interaction_version == resumed.interaction.interaction_version + 1
    assert later.interaction.request_ref != opened.interaction.request_ref
    assert store.get_job(context.job_spec.job_id).status is JobStatus.BLOCKED


def test_request_more_successor_requires_materialized_revision_and_supersedes(
    tmp_path: Path,
) -> None:
    store, service, context, opened = _opened_label(
        tmp_path,
        key="open-r7-07-revision-successor",
    )
    assert opened.interaction is not None
    waiting = service.decide(
        interaction_id=opened.interaction.interaction_id,
        submission=UserCheckpointDecisionSubmissionV2(
            interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
            decision=UserDecision.REQUEST_MORE_EXAMPLES,
            reason="Add examples before deciding.",
            idempotency_key="decide-r7-07-revision-successor",
        ),
        authentication=_authentication(),
        expected_job_version=opened.job_version,
        audit=_audit(),
    )
    pending_revision = waiting.decision_commit.request_revision
    assert pending_revision is not None
    successor_context = _successor_label_context(
        context,
        boundary="Use current tool facts and the new examples.",
    )

    with pytest.raises(
        UserCheckpointInteractionConflictError,
        match="materialized",
    ):
        service.open_checkpoint(
            source_context=successor_context,
            policy=_interaction_policy(),
            expected_job_version=waiting.job_version,
            idempotency_key="open-r7-07-unmaterialized-successor",
            audit=_audit(),
        )

    UserDecisionPersistenceService(store).materialize_request_revision(
        pending_revision=pending_revision,
        successor_request=successor_context.request_compilation.requests[0],
        handling_policy=context.handling_policy,
        idempotency_key="materialize-r7-07-successor",
        audit=_audit(),
    )
    successor = service.open_checkpoint(
        source_context=successor_context,
        policy=_interaction_policy(),
        expected_job_version=waiting.job_version,
        idempotency_key="open-r7-07-materialized-successor",
        audit=_audit(),
    )

    assert successor.interaction is not None
    history = service.list_interactions(
        context.job_spec.job_id,
        offset=0,
        limit=100,
    ).interactions
    assert tuple(value.state for value in history[-2:]) == (
        UserCheckpointInteractionStateV2.SUPERSEDED,
        UserCheckpointInteractionStateV2.PENDING_DECISION,
    )
    assert successor.interaction.predecessor_interaction_ref == (
        user_checkpoint_interaction_v2_ref(history[-2])
    )
    assert successor.job_version == waiting.job_version
    assert store.get_job(context.job_spec.job_id).status is JobStatus.BLOCKED


def test_deferred_final_successor_supersedes_without_reopening_job(
    tmp_path: Path,
) -> None:
    approval_policy = _policy(
        ApprovalMode.FINAL_ONLY,
        scope=FinalReviewScope.PROMPTS,
    )
    source = _final_source(FinalReviewScope.PROMPTS, sample_count=1)
    compilation = _compile(
        policy=approval_policy,
        checkpoint=ApprovalCheckpoint.FINAL_DATASET_REVIEW,
        sources=(source,),
    )
    store, service = _service(tmp_path, policy=approval_policy)
    running = store.transition_job(
        compilation.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-r7-07-defer-successor",
    )
    context = UserCheckpointSourceContextV2.create(
        job_spec=_job_spec(approval_policy),
        approval_policy=approval_policy,
        generation_policy=_generation_policy(),
        handling_policy=_handling_policy(),
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=(source,),
    )
    opened = service.open_checkpoint(
        source_context=context,
        policy=_interaction_policy(),
        expected_job_version=running.row_version,
        idempotency_key="open-r7-07-defer-successor",
        audit=_audit(),
    )
    assert opened.interaction is not None
    deferred = service.decide(
        interaction_id=opened.interaction.interaction_id,
        submission=UserCheckpointDecisionSubmissionV2(
            interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
            decision=UserDecision.DEFER,
            reason="Defer until the expanded sample is ready.",
            idempotency_key="decide-r7-07-defer-successor",
        ),
        authentication=_authentication(),
        expected_job_version=opened.job_version,
        audit=_audit(),
    )
    disabled_compilation = _compile(
        policy=approval_policy,
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        sources=(),
    )
    disabled_context = UserCheckpointSourceContextV2.create(
        job_spec=context.job_spec,
        approval_policy=approval_policy,
        generation_policy=context.generation_policy,
        handling_policy=context.handling_policy,
        request_compilation=disabled_compilation,
        requested_by=context.requested_by,
        sources=(),
    )
    with pytest.raises(
        UserCheckpointInteractionConflictError,
        match="successor",
    ):
        service.open_checkpoint(
            source_context=disabled_context,
            policy=_interaction_policy(),
            expected_job_version=deferred.job_version,
            idempotency_key="open-r7-07-disabled-while-deferred",
            audit=_audit(),
        )
    successor_source = _final_source(
        FinalReviewScope.PROMPTS,
        sample_count=2,
    )
    successor_compilation = _compile(
        policy=approval_policy,
        checkpoint=ApprovalCheckpoint.FINAL_DATASET_REVIEW,
        sources=(successor_source,),
    )
    successor_context = UserCheckpointSourceContextV2.create(
        job_spec=context.job_spec,
        approval_policy=approval_policy,
        generation_policy=context.generation_policy,
        handling_policy=context.handling_policy,
        request_compilation=successor_compilation,
        requested_by=context.requested_by,
        sources=(successor_source,),
    )

    successor = service.open_checkpoint(
        source_context=successor_context,
        policy=_interaction_policy(),
        expected_job_version=deferred.job_version,
        idempotency_key="open-r7-07-deferred-successor",
        audit=_audit(),
    )

    assert successor.interaction is not None
    history = service.list_interactions(
        context.job_spec.job_id,
        offset=0,
        limit=100,
    ).interactions
    assert tuple(value.state for value in history[-2:]) == (
        UserCheckpointInteractionStateV2.SUPERSEDED,
        UserCheckpointInteractionStateV2.PENDING_DECISION,
    )
    assert successor.job_version == deferred.job_version
    assert store.get_job(context.job_spec.job_id).status is JobStatus.BLOCKED


def test_authentication_and_nonresumable_states_fail_closed(
    tmp_path: Path,
) -> None:
    _, service, context, opened = _opened_label(
        tmp_path,
        key="open-r7-07-auth",
    )
    assert opened.interaction is not None
    with pytest.raises(UserCheckpointAuthenticationError):
        service.decide(
            interaction_id=opened.interaction.interaction_id,
            submission=UserCheckpointDecisionSubmissionV2(
                interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
                decision=UserDecision.ACCEPT,
                reason="Mismatched user.",
                idempotency_key="decide-r7-07-wrong-user",
            ),
            authentication=_authentication("another-user"),
            expected_job_version=opened.job_version,
            audit=_audit(),
        )
    waiting = service.decide(
        interaction_id=opened.interaction.interaction_id,
        submission=UserCheckpointDecisionSubmissionV2(
            interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
            decision=UserDecision.REQUEST_MORE_EXAMPLES,
            reason="Request more examples.",
            idempotency_key="decide-r7-07-more",
        ),
        authentication=_authentication(),
        expected_job_version=opened.job_version,
        audit=_audit(),
    )
    assert waiting.interaction.state is (UserCheckpointInteractionStateV2.WAITING_REVISION)
    with pytest.raises(UserCheckpointInteractionNotResumableError):
        service.resume_checkpoint(
            job_id=context.job_spec.job_id,
            expected_job_version=waiting.job_version,
            idempotency_key="resume-r7-07-more",
            audit=_audit(),
        )
    disabled_compilation = _compile(
        policy=context.approval_policy,
        checkpoint=ApprovalCheckpoint.FINAL_DATASET_REVIEW,
        sources=(),
    )
    disabled_context = UserCheckpointSourceContextV2.create(
        job_spec=context.job_spec,
        approval_policy=context.approval_policy,
        generation_policy=context.generation_policy,
        handling_policy=context.handling_policy,
        request_compilation=disabled_compilation,
        requested_by=context.requested_by,
        sources=(),
    )
    with pytest.raises(
        UserCheckpointInteractionConflictError,
        match="successor",
    ):
        service.open_checkpoint(
            source_context=disabled_context,
            policy=_interaction_policy(),
            expected_job_version=waiting.job_version,
            idempotency_key="open-r7-07-disabled-while-waiting",
            audit=_audit(),
        )


def test_final_reject_is_resumable_for_r7_08(
    tmp_path: Path,
) -> None:
    approval_policy = _policy(
        ApprovalMode.FINAL_ONLY,
        scope=FinalReviewScope.PROMPTS,
    )
    source = _final_source(FinalReviewScope.PROMPTS)
    compilation = _compile(
        policy=approval_policy,
        checkpoint=ApprovalCheckpoint.FINAL_DATASET_REVIEW,
        sources=(source,),
    )
    store, service = _service(tmp_path, policy=approval_policy)
    running = store.transition_job(
        compilation.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-r7-07-final-reject",
    )
    context = UserCheckpointSourceContextV2.create(
        job_spec=_job_spec(approval_policy),
        approval_policy=approval_policy,
        generation_policy=_generation_policy(),
        handling_policy=_handling_policy(),
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=(source,),
    )
    opened = service.open_checkpoint(
        source_context=context,
        policy=_interaction_policy(),
        expected_job_version=running.row_version,
        idempotency_key="open-r7-07-final-reject",
        audit=_audit(),
    )
    assert opened.interaction is not None
    rejected = service.decide(
        interaction_id=opened.interaction.interaction_id,
        submission=UserCheckpointDecisionSubmissionV2(
            interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
            decision=UserDecision.REJECT,
            reason="Reject final candidate.",
            idempotency_key="decide-r7-07-final-reject",
        ),
        authentication=_authentication(),
        expected_job_version=opened.job_version,
        audit=_audit(),
    )
    assert rejected.interaction.state is (UserCheckpointInteractionStateV2.RESUMABLE)
    assert rejected.resume_disposition is (UserCheckpointResumeDispositionV2.DIRECT)
    assert store.get_job(compilation.job_id).status is JobStatus.BLOCKED


def test_missing_current_head_fails_closed_and_rebuild_restores_it(
    tmp_path: Path,
) -> None:
    store, service, context, opened = _opened_label(
        tmp_path,
        key="open-r7-07-rebuild",
    )
    assert opened.interaction is not None
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            DELETE FROM user_checkpoint_current_heads
            WHERE job_id = ?
            """,
            (context.job_spec.job_id,),
        )
    with pytest.raises(ImmutableResultError, match="head is missing"):
        service.get_current_interaction(context.job_spec.job_id)
    with pytest.raises(ImmutableResultError, match="head is missing"):
        store.has_current_checkpoint_interaction(context.job_spec.job_id)
    assert service.rebuild_current_head(context.job_spec.job_id) == (opened.interaction)
    assert service.get_current_interaction(context.job_spec.job_id) == (opened.interaction)


def test_generic_resume_guard_rejects_synchronized_state_column_drift(
    tmp_path: Path,
) -> None:
    store, _, context, opened = _opened_label(
        tmp_path,
        key="open-r7-07-guard-drift",
    )
    assert opened.interaction is not None
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE user_checkpoint_interactions
            SET state = 'RESUMED'
            WHERE interaction_id = ?
            """,
            (opened.interaction.interaction_id,),
        )
        connection.execute(
            """
            UPDATE user_checkpoint_current_heads
            SET state = 'RESUMED'
            WHERE job_id = ?
            """,
            (context.job_spec.job_id,),
        )

    with pytest.raises(ImmutableResultError, match="latest checkpoint"):
        PipelineLifecycleService(store).resume(
            job_id=context.job_spec.job_id,
            expected_job_version=opened.job_version,
            idempotency_key="generic-resume-r7-07-guard-drift",
            offset=0,
            limit=100,
        )


def test_decision_target_result_type_and_current_head_drift_fail_closed(
    tmp_path: Path,
) -> None:
    store, service, context, opened = _opened_label(
        tmp_path,
        key="open-r7-07-drift",
    )
    assert opened.interaction is not None
    wrong_ref = user_checkpoint_interaction_v2_ref(opened.interaction).model_copy(
        update={"object_id": "user-checkpoint-interaction://wrong"}
    )
    with pytest.raises(
        UserCheckpointInteractionConflictError,
        match="another",
    ):
        service.decide(
            interaction_id=opened.interaction.interaction_id,
            submission=UserCheckpointDecisionSubmissionV2(
                interaction_ref=wrong_ref,
                decision=UserDecision.ACCEPT,
                reason="Wrong interaction.",
                idempotency_key="decide-r7-07-wrong-ref",
            ),
            authentication=_authentication(),
            expected_job_version=opened.job_version,
            audit=_audit(),
        )
    accepted = service.decide(
        interaction_id=opened.interaction.interaction_id,
        submission=UserCheckpointDecisionSubmissionV2(
            interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
            decision=UserDecision.ACCEPT,
            reason="Accept once.",
            idempotency_key="decide-r7-07-once",
        ),
        authentication=_authentication(),
        expected_job_version=opened.job_version,
        audit=_audit(),
    )
    with pytest.raises(
        UserCheckpointInteractionConflictError,
        match="current pending",
    ):
        service.decide(
            interaction_id=opened.interaction.interaction_id,
            submission=UserCheckpointDecisionSubmissionV2(
                interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
                decision=UserDecision.ACCEPT,
                reason="Accept twice.",
                idempotency_key="decide-r7-07-twice",
            ),
            authentication=_authentication(),
            expected_job_version=opened.job_version,
            audit=_audit(),
        )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE user_checkpoint_results
            SET result_type = 'UNKNOWN'
            WHERE result_id = ?
            """,
            (accepted.result_id,),
        )
    with pytest.raises(ImmutableResultError, match="unsupported"):
        service.get_result(accepted.result_id)

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE user_checkpoint_current_heads
            SET state = 'DEFERRED'
            WHERE job_id = ?
            """,
            (context.job_spec.job_id,),
        )
    with pytest.raises(ImmutableResultError, match="current head"):
        service.get_current_interaction(context.job_spec.job_id)


def test_interaction_row_corruption_and_page_bounds_fail_closed(
    tmp_path: Path,
) -> None:
    store, service, context, opened = _opened_label(
        tmp_path,
        key="open-r7-07-row-corrupt",
    )
    assert opened.interaction is not None
    with pytest.raises(
        UserCheckpointInteractionError,
        match="page",
    ):
        service.list_interactions(
            context.job_spec.job_id,
            offset=-1,
            limit=100,
        )
    with pytest.raises(RecordNotFoundError):
        service.get_result("user-checkpoint-result://missing")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE user_checkpoint_interactions
            SET record_json = '{}'
            WHERE interaction_id = ?
            """,
            (opened.interaction.interaction_id,),
        )
    with pytest.raises(ImmutableResultError, match="malformed"):
        service.get_interaction(opened.interaction.interaction_id)


def test_interaction_predecessor_chain_drift_fails_closed(
    tmp_path: Path,
) -> None:
    store, service, context, opened = _opened_label(
        tmp_path,
        key="open-r7-07-chain-drift",
    )
    assert opened.interaction is not None
    accepted = service.decide(
        interaction_id=opened.interaction.interaction_id,
        submission=UserCheckpointDecisionSubmissionV2(
            interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
            decision=UserDecision.ACCEPT,
            reason="Accept before chain drift.",
            idempotency_key="decide-r7-07-chain-drift",
        ),
        authentication=_authentication(),
        expected_job_version=opened.job_version,
        audit=_audit(),
    )
    wrong_predecessor = user_checkpoint_interaction_v2_ref(opened.interaction).model_copy(
        update={
            "object_id": "user-checkpoint-interaction://wrong-predecessor",
            "object_sha256": "f" * 64,
        }
    )
    tampered = UserCheckpointInteractionV2.create(
        job_id=accepted.interaction.job_id,
        chain_id=accepted.interaction.chain_id,
        interaction_version=accepted.interaction.interaction_version,
        predecessor_interaction_ref=wrong_predecessor,
        checkpoint=accepted.interaction.checkpoint,
        request_compilation_ref=accepted.interaction.request_compilation_ref,
        request_ref=accepted.interaction.request_ref,
        presentation_ref=accepted.interaction.presentation_ref,
        source_context_ref=accepted.interaction.source_context_ref,
        state=accepted.interaction.state,
        decision_commit_ref=accepted.interaction.decision_commit_ref,
        application_ref=accepted.interaction.application_ref,
        job_status=accepted.interaction.job_status,
        occurred_at=accepted.interaction.occurred_at,
        policy_ref=accepted.interaction.policy_ref,
        audit=_audit(),
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE user_checkpoint_interactions
            SET interaction_id = ?, interaction_sha256 = ?, record_json = ?
            WHERE interaction_id = ?
            """,
            (
                tampered.interaction_id,
                tampered.interaction_sha256,
                tampered.model_dump_json(),
                accepted.interaction.interaction_id,
            ),
        )
        connection.execute(
            """
            UPDATE user_checkpoint_current_heads
            SET interaction_id = ?
            WHERE job_id = ?
            """,
            (
                tampered.interaction_id,
                context.job_spec.job_id,
            ),
        )
    with pytest.raises(ImmutableResultError, match="chain"):
        service.get_current_interaction(context.job_spec.job_id)


def test_missing_head_offset_and_nonrunning_open_are_rejected(
    tmp_path: Path,
) -> None:
    approval_policy, source, compilation, _ = _label_request()
    _, service = _service(tmp_path, policy=approval_policy)
    with pytest.raises(RecordNotFoundError, match="current"):
        service.get_current_interaction(compilation.job_id)
    with pytest.raises(UserCheckpointInteractionError, match="offset"):
        service.list_interactions(
            compilation.job_id,
            offset=1,
            limit=100,
        )
    context = UserCheckpointSourceContextV2.create(
        job_spec=_job_spec(approval_policy),
        approval_policy=approval_policy,
        generation_policy=_generation_policy(),
        handling_policy=_handling_policy(),
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=(source,),
    )
    with pytest.raises(
        UserCheckpointInteractionConflictError,
        match="RUNNING",
    ):
        service.open_checkpoint(
            source_context=context,
            policy=_interaction_policy(),
            expected_job_version=0,
            idempotency_key="open-r7-07-created",
            audit=_audit(),
        )


def test_decision_cannot_exceed_interaction_policy_limit(
    tmp_path: Path,
) -> None:
    approval_policy, source, compilation, _ = _label_request()
    store, service = _service(tmp_path, policy=approval_policy)
    running = store.transition_job(
        compilation.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-r7-07-interaction-limit",
    )
    context = UserCheckpointSourceContextV2.create(
        job_spec=_job_spec(approval_policy),
        approval_policy=approval_policy,
        generation_policy=_generation_policy(),
        handling_policy=_handling_policy(),
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=(source,),
    )
    limited_policy = UserCheckpointInteractionPolicyV2.create(
        max_interactions_per_job=1,
        max_requests_per_interaction=10,
        max_page_size=100,
        max_presentation_bytes=1_000_000,
        max_source_context_bytes=2_000_000,
        max_ready_work_refs=100,
        audit=_audit(),
    )
    opened = service.open_checkpoint(
        source_context=context,
        policy=limited_policy,
        expected_job_version=running.row_version,
        idempotency_key="open-r7-07-interaction-limit",
        audit=_audit(),
    )
    assert opened.interaction is not None

    with pytest.raises(
        UserCheckpointInteractionError,
        match="interaction count",
    ):
        service.decide(
            interaction_id=opened.interaction.interaction_id,
            submission=UserCheckpointDecisionSubmissionV2(
                interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
                decision=UserDecision.ACCEPT,
                reason="This decision exceeds the interaction limit.",
                idempotency_key="decide-r7-07-interaction-limit",
            ),
            authentication=_authentication(),
            expected_job_version=opened.job_version,
            audit=_audit(),
        )
    assert service.get_current_interaction(compilation.job_id) == (opened.interaction)


def test_next_request_cannot_exceed_interaction_policy_limit(
    tmp_path: Path,
) -> None:
    approval_policy = _policy(ApprovalMode.PLAN_GATES)
    sources = (
        _label_source("limit-first"),
        _label_source("limit-second"),
    )
    compilation = _compile(
        policy=approval_policy,
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        sources=sources,
    )
    store, service = _service(tmp_path, policy=approval_policy)
    running = store.transition_job(
        compilation.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-r7-07-next-limit",
    )
    context = UserCheckpointSourceContextV2.create(
        job_spec=_job_spec(approval_policy),
        approval_policy=approval_policy,
        generation_policy=_generation_policy(),
        handling_policy=_handling_policy(),
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=sources,
    )
    limited_policy = UserCheckpointInteractionPolicyV2.create(
        max_interactions_per_job=2,
        max_requests_per_interaction=10,
        max_page_size=100,
        max_presentation_bytes=1_000_000,
        max_source_context_bytes=2_000_000,
        max_ready_work_refs=100,
        audit=_audit(),
    )
    opened = service.open_checkpoint(
        source_context=context,
        policy=limited_policy,
        expected_job_version=running.row_version,
        idempotency_key="open-r7-07-next-limit",
        audit=_audit(),
    )
    assert opened.interaction is not None

    with pytest.raises(
        UserCheckpointInteractionError,
        match="interaction count",
    ):
        service.decide(
            interaction_id=opened.interaction.interaction_id,
            submission=UserCheckpointDecisionSubmissionV2(
                interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
                decision=UserDecision.ACCEPT,
                reason="The next request exceeds the interaction limit.",
                idempotency_key="decide-r7-07-next-limit",
            ),
            authentication=_authentication(),
            expected_job_version=opened.job_version,
            audit=_audit(),
        )
    assert service.get_current_interaction(compilation.job_id) == (opened.interaction)
    assert (
        service.list_interactions(
            compilation.job_id,
            offset=0,
            limit=100,
        ).total
        == 1
    )


def test_adjust_resume_requires_and_returns_r7_06_ready_work(
    tmp_path: Path,
) -> None:
    approval_policy, approval_source, compilation, request = _label_request()
    job_spec = _job_spec(approval_policy)
    store = JobStore(tmp_path / "adjust.sqlite3")
    store.create_planned_job(job_spec, _plan(job_spec))
    graph = _work_graph(job_spec)
    store.create_job_work_graph(
        graph,
        idempotency_key="create-r7-07-adjust-graph",
    )
    service = UserCheckpointInteractionService(
        store,
        UserCheckpointMaterialStore(
            tmp_path / "checkpoint-store",
            max_source_context_bytes=2_000_000,
            max_presentation_bytes=1_000_000,
        ),
    )
    running = store.transition_job(
        job_spec.job_id,
        JobStatus.RUNNING,
        expected_version=0,
        idempotency_key="start-r7-07-adjust",
    )
    context = UserCheckpointSourceContextV2.create(
        job_spec=job_spec,
        approval_policy=approval_policy,
        generation_policy=_generation_policy(),
        handling_policy=_handling_policy(),
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=(approval_source,),
    )
    opened = service.open_checkpoint(
        source_context=context,
        policy=_interaction_policy(),
        expected_job_version=running.row_version,
        idempotency_key="open-r7-07-adjust",
        audit=_audit(),
    )
    assert opened.interaction is not None
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
    decided = service.decide(
        interaction_id=opened.interaction.interaction_id,
        submission=UserCheckpointDecisionSubmissionV2(
            interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
            decision=UserDecision.ADJUST,
            adjustments=(adjustment,),
            adjustment_effect=producer.effect,
            reason="Apply the tightened label boundary.",
            idempotency_key="decide-r7-07-adjust",
        ),
        authentication=_authentication(),
        expected_job_version=opened.job_version,
        audit=_audit(),
    )
    with pytest.raises(UserCheckpointInteractionNotResumableError):
        service.resume_checkpoint(
            job_id=job_spec.job_id,
            expected_job_version=decided.job_version,
            idempotency_key="resume-r7-07-adjust-too-early",
            audit=_audit(),
        )
    item_sources = (
        RevalidationItemSource(
            item_id=graph.item_ids[0],
            source_trace_ref=graph.source_trace_refs[0],
            authority_refs=(request.subject_refs[0],),
            stages=(
                RevalidationStageSource(
                    stage=StageNameV2.LABEL,
                    output_refs=(_ref("label-decision", "r7-07-source"),),
                ),
                RevalidationStageSource(
                    stage=StageNameV2.TASK_AUTHORING,
                    output_refs=(_ref("r4-task-contract-set", "r7-07-source"),),
                ),
                RevalidationStageSource(
                    stage=StageNameV2.ATTACHMENT,
                    output_refs=(
                        _ref(
                            "attachment-reconstruction-result",
                            "r7-07-source",
                        ),
                    ),
                ),
                RevalidationStageSource(
                    stage=StageNameV2.ITEM_QUALITY,
                    output_refs=(
                        _ref(
                            "item-quality-compilation-result",
                            "r7-07-source",
                        ),
                    ),
                ),
            ),
        ),
    )
    application_service = UserPlanApplicationPersistenceService(store)
    application = application_service.apply(
        decision_commit_id=decided.decision_commit.commit_id,
        producer_result=producer.result,
        item_sources=item_sources,
        policy=_application_policy(),
        idempotency_key="apply-r7-07-adjust",
        audit=_audit(),
    )
    resumed = service.resume_checkpoint(
        job_id=job_spec.job_id,
        expected_job_version=decided.job_version,
        idempotency_key="resume-r7-07-adjust",
        audit=_audit(),
    )
    assert resumed.application_ref is not None
    assert resumed.application_ref.object_id == (application.application.application_id)
    assert resumed.revalidation_ready_work_refs
    assert resumed.incomplete_work_refs == ()
    assert store.get_job(job_spec.job_id).status is JobStatus.RUNNING


def test_historical_resumed_head_does_not_block_ordinary_pipeline_resume(
    tmp_path: Path,
) -> None:
    approval_policy, source, compilation, _ = _label_request()
    job_spec = _job_spec(approval_policy)
    store = JobStore(tmp_path / "historical-resumed.sqlite3")
    pipeline = PipelineLifecycleService(store)
    created = pipeline.create(
        job_spec=job_spec,
        control=PipelineControlConfigV2(
            lease_duration_seconds=30,
            heartbeat_extension_seconds=20,
            max_attempts=2,
            retry_delay_seconds=(5,),
            retry_lease_expiry=True,
        ),
        audit=_audit(),
        idempotency_key=job_spec.idempotency_key,
    )
    running = pipeline.resume(
        job_id=job_spec.job_id,
        expected_job_version=created.job_version,
        idempotency_key="start-r7-07-historical-resumed",
        offset=0,
        limit=100,
    )
    service = UserCheckpointInteractionService(
        store,
        UserCheckpointMaterialStore(
            tmp_path / "historical-resumed-material",
            max_source_context_bytes=2_000_000,
            max_presentation_bytes=1_000_000,
        ),
    )
    context = UserCheckpointSourceContextV2.create(
        job_spec=job_spec,
        approval_policy=approval_policy,
        generation_policy=_generation_policy(),
        handling_policy=_handling_policy(),
        request_compilation=compilation,
        requested_by="requesting-user",
        sources=(source,),
    )
    opened = service.open_checkpoint(
        source_context=context,
        policy=_interaction_policy(),
        expected_job_version=running.status.job_version,
        idempotency_key="open-r7-07-historical-resumed",
        audit=_audit(),
    )
    assert opened.interaction is not None
    decided = service.decide(
        interaction_id=opened.interaction.interaction_id,
        submission=UserCheckpointDecisionSubmissionV2(
            interaction_ref=user_checkpoint_interaction_v2_ref(opened.interaction),
            decision=UserDecision.ACCEPT,
            reason="Accept before the ordinary block.",
            idempotency_key="decide-r7-07-historical-resumed",
        ),
        authentication=_authentication(),
        expected_job_version=opened.job_version,
        audit=_audit(),
    )
    checkpoint_resume = service.resume_checkpoint(
        job_id=job_spec.job_id,
        expected_job_version=decided.job_version,
        idempotency_key="resume-r7-07-historical-resumed",
        audit=_audit(),
    )
    assert checkpoint_resume.incomplete_work_refs
    blocked = store.transition_job(
        job_spec.job_id,
        JobStatus.BLOCKED,
        expected_version=checkpoint_resume.job_version,
        idempotency_key="ordinary-block-after-checkpoint",
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            DELETE FROM user_checkpoint_current_heads
            WHERE job_id = ?
            """,
            (job_spec.job_id,),
        )
    assert service.rebuild_current_head(job_spec.job_id) == (checkpoint_resume.interaction)

    reopened = pipeline.resume(
        job_id=job_spec.job_id,
        expected_job_version=blocked.row_version,
        idempotency_key="ordinary-resume-after-checkpoint",
        offset=0,
        limit=100,
    )

    assert reopened.action is PipelineResumeActionV2.REOPENED
