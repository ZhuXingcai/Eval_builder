from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import ValidationError

from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    EnvironmentStrategyChoice,
    InvalidationScope,
    TypedAdjustment,
)
from eval_factory.contracts.approval_application_v2 import (
    DirectedRevalidationPlanV2,
    DirectedRevalidationReportOutcomeV2,
    DirectedRevalidationReportV2,
    EnvironmentStrategyAdjustmentResultV2,
    FinalDatasetAdjustmentResultV2,
    LabelPlanAdjustmentResultV2,
    RevalidationWorkItemV2,
    RevalidationWorkOutcomeV2,
    RevalidationWorkResultV2,
    RevalidationWorkScopeV2,
    UserPlanApplicationPolicyV2,
    UserPlanApplicationV2,
    directed_revalidation_plan_v2_ref,
    environment_strategy_adjustment_result_v2_ref,
    final_dataset_adjustment_result_v2_ref,
    label_plan_adjustment_result_v2_ref,
    revalidation_work_item_v2_ref,
    user_plan_application_policy_v2_ref,
    user_plan_application_v2_ref,
    validate_directed_revalidation_plan_v2_identity,
    validate_directed_revalidation_report_v2_identity,
    validate_environment_strategy_adjustment_result_v2_identity,
    validate_final_dataset_adjustment_result_v2_identity,
    validate_label_plan_adjustment_result_v2_identity,
    validate_revalidation_work_item_v2_identity,
    validate_revalidation_work_result_v2_identity,
    validate_user_plan_application_policy_v2_identity,
    validate_user_plan_application_v2_identity,
)
from eval_factory.contracts.approval_decision_v2 import (
    UserDecisionAdjustmentEffectV2,
    UserDecisionCommitOutcomeV2,
    UserDecisionCommitResultV2,
    user_decision_adjustment_effect_v2_ref,
    user_decision_commit_result_v2_ref,
    validate_user_decision_commit_result_v2_identity,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.orchestration import StageRunStatus
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    ResolvedJobWorkGraphV2,
    StageNameV2,
    dataset_job_spec_v2_ref,
    resolved_job_work_graph_v2_ref,
    validate_resolved_job_work_graph_v2_identity,
)
from eval_factory.contracts.task_v2 import (
    r4_task_contract_set_carried_sha256,
    r4_task_contract_set_ref,
    task_contract_invalidation_carried_sha256,
    task_contract_invalidation_ref,
    task_rewrite_application_carried_sha256,
    task_rewrite_application_ref,
    task_rewrite_plan_preview_carried_sha256,
    task_rewrite_plan_preview_ref,
    task_rewrite_plan_version_carried_sha256,
    task_rewrite_plan_version_ref,
    task_rewrite_preview_safety_gate_carried_sha256,
    task_rewrite_preview_safety_gate_ref,
)
from eval_factory.orchestration.models import (
    StageResultRecord,
    StageRunRecord,
    stage_result_record_carried_sha256,
    stage_result_record_ref,
)
from eval_factory.task_authoring.rewrite_models import (
    TaskRewriteAdjustmentResult,
    TaskRewriteApplicationResult,
    TaskRewritePlanCompileOutcome,
    rewrite_payload_sha256,
)


class UserPlanApplicationPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RevalidationStageSource:
    stage: StageNameV2
    output_refs: tuple[ObjectRef, ...]


@dataclass(frozen=True, slots=True)
class RevalidationItemSource:
    item_id: str
    source_trace_ref: ObjectRef
    authority_refs: tuple[ObjectRef, ...]
    stages: tuple[RevalidationStageSource, ...]
    environment_requirement_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TaskRewriteApplicationSource:
    adjustment_result: TaskRewriteAdjustmentResult
    application_result: TaskRewriteApplicationResult


type AdjustmentProducerResult = (
    LabelPlanAdjustmentResultV2
    | EnvironmentStrategyAdjustmentResultV2
    | FinalDatasetAdjustmentResultV2
    | TaskRewriteApplicationSource
)


@dataclass(frozen=True, slots=True)
class UserPlanApplicationCompilation:
    application: UserPlanApplicationV2
    plan: DirectedRevalidationPlanV2


@dataclass(frozen=True, slots=True)
class _ProducerAuthority:
    source_root_refs: tuple[ObjectRef, ...]
    replacement_root_refs: tuple[ObjectRef, ...]
    invalidated_root_refs: tuple[ObjectRef, ...]
    affected_item_ids: tuple[str, ...]
    excluded_item_ids: tuple[str, ...]
    restart_by_item: tuple[tuple[str, StageNameV2], ...]


_ITEM_EXECUTION_STAGES = (
    StageNameV2.LABEL,
    StageNameV2.TASK_AUTHORING,
    StageNameV2.ATTACHMENT,
    StageNameV2.ITEM_QUALITY,
)
_EXECUTION_STAGES = (
    *_ITEM_EXECUTION_STAGES,
    StageNameV2.BATCH_QUALITY,
    StageNameV2.RELEASE,
)
_OUTPUT_TYPES = {
    StageNameV2.LABEL: ("label-decision",),
    StageNameV2.TASK_AUTHORING: ("r4-task-contract-set",),
    StageNameV2.ATTACHMENT: ("attachment-reconstruction-result",),
    StageNameV2.ITEM_QUALITY: ("item-quality-compilation-result",),
    StageNameV2.BATCH_QUALITY: ("batch-quality-report",),
}
_HARD_GATE_STAGES = frozenset(
    {
        StageNameV2.TASK_AUTHORING,
        StageNameV2.ATTACHMENT,
        StageNameV2.ITEM_QUALITY,
        StageNameV2.BATCH_QUALITY,
    }
)
_DOWNSTREAM_REAPPROVALS = {
    ApprovalCheckpoint.LABEL_PLAN: (
        ApprovalCheckpoint.TASK_REWRITE_PLAN,
        ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        ApprovalCheckpoint.FINAL_DATASET_REVIEW,
    ),
    ApprovalCheckpoint.TASK_REWRITE_PLAN: (
        ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        ApprovalCheckpoint.FINAL_DATASET_REVIEW,
    ),
    ApprovalCheckpoint.ENVIRONMENT_STRATEGY: (ApprovalCheckpoint.FINAL_DATASET_REVIEW,),
    ApprovalCheckpoint.FINAL_DATASET_REVIEW: (ApprovalCheckpoint.FINAL_DATASET_REVIEW,),
}


class UserPlanApplicationCompiler:
    def compile(
        self,
        *,
        decision_commit: UserDecisionCommitResultV2,
        producer_result: AdjustmentProducerResult,
        job_spec: DatasetJobSpecV2,
        resolved_job_work_graph: ResolvedJobWorkGraphV2,
        item_sources: tuple[RevalidationItemSource, ...],
        policy: UserPlanApplicationPolicyV2,
        audit: ContractAudit,
        job_stage_sources: tuple[RevalidationStageSource, ...] = (),
    ) -> UserPlanApplicationCompilation:
        commit, job, graph, current_policy = _admit_application_sources(
            decision_commit=decision_commit,
            job_spec=job_spec,
            resolved_job_work_graph=resolved_job_work_graph,
            policy=policy,
        )
        sources = _admit_item_sources(item_sources, graph)
        job_sources = _admit_stage_sources(
            job_stage_sources,
            label="Job revalidation sources",
        )
        effect = commit.adjustment_effect
        assert effect is not None
        authority = _admit_producer(
            commit=commit,
            producer_result=producer_result,
            item_sources=sources,
        )
        restart_by_item = dict(authority.restart_by_item)
        restart_stages = _application_restart_stages(restart_by_item)
        invalidated_refs = _invalidated_refs(
            commit=commit,
            producer_result=producer_result,
            authority=authority,
            item_sources=sources,
            job_stage_sources=job_sources,
            restart_by_item=restart_by_item,
        )
        preserved_refs = _preserved_refs(
            authority=authority,
            item_sources=sources,
            job_stage_sources=job_sources,
            restart_by_item=restart_by_item,
            invalidated_refs=invalidated_refs,
        )
        reapprovals = tuple(
            checkpoint
            for checkpoint in _DOWNSTREAM_REAPPROVALS[effect.checkpoint]
            if checkpoint in job.enabled_checkpoints
        )
        _preflight_application_budgets(
            policy=current_policy,
            adjustment_count=len(effect.adjustments),
            invalidated_refs=invalidated_refs,
            preserved_refs=preserved_refs,
            affected_item_ids=authority.affected_item_ids,
        )
        application = UserPlanApplicationV2.create(
            job_id=job.job_id,
            decision_commit_ref=user_decision_commit_result_v2_ref(commit),
            decision_record_ref=commit.decision_record_ref,
            adjustment_effect_ref=user_decision_adjustment_effect_v2_ref(effect),
            checkpoint=effect.checkpoint,
            source_root_refs=authority.source_root_refs,
            replacement_root_refs=authority.replacement_root_refs,
            invalidated_object_refs=invalidated_refs,
            preserved_object_refs=preserved_refs,
            affected_item_ids=authority.affected_item_ids,
            excluded_item_ids=authority.excluded_item_ids,
            restart_stages=restart_stages,
            required_checkpoint_reapprovals=reapprovals,
            policy_ref=user_plan_application_policy_v2_ref(current_policy),
            audit=audit,
        )
        work_items = _compile_work_items(
            application=application,
            authority=authority,
            item_sources=sources,
            job_stage_sources=job_sources,
            restart_by_item=restart_by_item,
        )
        if len(work_items) > current_policy.max_work_items:
            raise UserPlanApplicationPolicyError("directed revalidation exceeds the work item limit")
        plan = DirectedRevalidationPlanV2.create(
            application_ref=user_plan_application_v2_ref(application),
            resolved_job_work_graph_ref=resolved_job_work_graph_v2_ref(graph),
            work_items=work_items,
            preserved_item_ids=tuple(
                source.item_id for source in sources if source.item_id not in set(authority.affected_item_ids)
            ),
            excluded_item_ids=authority.excluded_item_ids,
            audit=audit,
        )
        return UserPlanApplicationCompilation(application=application, plan=plan)

    def validate_current(
        self,
        compilation: UserPlanApplicationCompilation,
        *,
        decision_commit: UserDecisionCommitResultV2,
        producer_result: AdjustmentProducerResult,
        job_spec: DatasetJobSpecV2,
        resolved_job_work_graph: ResolvedJobWorkGraphV2,
        item_sources: tuple[RevalidationItemSource, ...],
        policy: UserPlanApplicationPolicyV2,
        job_stage_sources: tuple[RevalidationStageSource, ...] = (),
    ) -> None:
        try:
            validate_user_plan_application_v2_identity(compilation.application)
            validate_directed_revalidation_plan_v2_identity(compilation.plan)
        except (ValidationError, ValueError) as exc:
            raise UserPlanApplicationPolicyError("user plan application is stale or malformed") from exc
        rebuilt = self.compile(
            decision_commit=decision_commit,
            producer_result=producer_result,
            job_spec=job_spec,
            resolved_job_work_graph=resolved_job_work_graph,
            item_sources=item_sources,
            policy=policy,
            audit=compilation.application.audit,
            job_stage_sources=job_stage_sources,
        )
        if user_plan_application_v2_ref(rebuilt.application) != user_plan_application_v2_ref(
            compilation.application
        ) or directed_revalidation_plan_v2_ref(rebuilt.plan) != directed_revalidation_plan_v2_ref(
            compilation.plan
        ):
            raise UserPlanApplicationPolicyError("user plan application does not match current authority")


class DirectedRevalidationCompiler:
    def record_result(
        self,
        *,
        application: UserPlanApplicationV2,
        plan: DirectedRevalidationPlanV2,
        work_item: RevalidationWorkItemV2,
        stage_run: StageRunRecord,
        stage_result: StageResultRecord,
        dependency_results: tuple[RevalidationWorkResultV2, ...],
        policy: UserPlanApplicationPolicyV2,
        audit: ContractAudit,
    ) -> RevalidationWorkResultV2:
        try:
            current_application = UserPlanApplicationV2.model_validate(application.model_dump(mode="python"))
            current_plan = DirectedRevalidationPlanV2.model_validate(plan.model_dump(mode="python"))
            current_work = RevalidationWorkItemV2.model_validate(work_item.model_dump(mode="python"))
            current_run = StageRunRecord.model_validate(stage_run.model_dump(mode="python"))
            current_result = StageResultRecord.model_validate(stage_result.model_dump(mode="python"))
            current_policy = UserPlanApplicationPolicyV2.model_validate(policy.model_dump(mode="python"))
            validate_user_plan_application_v2_identity(current_application)
            validate_directed_revalidation_plan_v2_identity(current_plan)
            validate_revalidation_work_item_v2_identity(current_work)
            validate_user_plan_application_policy_v2_identity(current_policy)
        except (ValidationError, ValueError) as exc:
            raise UserPlanApplicationPolicyError("revalidation result source is stale or malformed") from exc
        application_ref = user_plan_application_v2_ref(current_application)
        work_ref = revalidation_work_item_v2_ref(current_work)
        if (
            current_plan.application_ref != application_ref
            or current_work.application_ref != application_ref
            or current_work not in current_plan.work_items
        ):
            raise UserPlanApplicationPolicyError("revalidation work is not a member of the application plan")
        _validate_dependency_results(
            work=current_work,
            dependency_results=dependency_results,
        )
        if (
            current_run.job_id != current_application.job_id
            or current_run.item_id != current_work.item_id
            or current_run.stage is not current_work.stage
            or current_run.input_refs != (work_ref,)
        ):
            raise UserPlanApplicationPolicyError("StageRun does not witness the exact revalidation work")
        if current_run.attempt > current_policy.max_revalidation_attempts:
            raise UserPlanApplicationPolicyError("StageRun exceeds the revalidation attempt limit")
        if (
            current_result.stage_run_id != current_run.stage_run_id
            or current_result.stage_run_version != current_run.row_version
            or current_result.status is not current_run.status
            or current_run.ended_at is None
        ):
            raise UserPlanApplicationPolicyError("StageResult does not witness the terminal StageRun")
        if current_result.result_sha256 != stage_result_record_carried_sha256(current_result):
            raise UserPlanApplicationPolicyError("StageResult identity is stale")
        outcome, outputs, failure_code = _result_outcome(
            work=current_work,
            stage_result=current_result,
            policy=current_policy,
            invalidated_refs=current_application.invalidated_object_refs,
        )
        return RevalidationWorkResultV2.create(
            work_item_ref=work_ref,
            stage_result_ref=stage_result_record_ref(current_result),
            outcome=outcome,
            output_refs=outputs,
            failure_code=failure_code,
            audit=audit,
        )

    def compile_report(
        self,
        *,
        application: UserPlanApplicationV2,
        plan: DirectedRevalidationPlanV2,
        work_results: tuple[RevalidationWorkResultV2, ...],
        audit: ContractAudit,
    ) -> DirectedRevalidationReportV2:
        try:
            current_application = UserPlanApplicationV2.model_validate(application.model_dump(mode="python"))
            current_plan = DirectedRevalidationPlanV2.model_validate(plan.model_dump(mode="python"))
            validate_user_plan_application_v2_identity(current_application)
            validate_directed_revalidation_plan_v2_identity(current_plan)
        except (ValidationError, ValueError) as exc:
            raise UserPlanApplicationPolicyError("revalidation report source is stale or malformed") from exc
        application_ref = user_plan_application_v2_ref(current_application)
        if current_plan.application_ref != application_ref:
            raise UserPlanApplicationPolicyError("revalidation plan belongs to another application")
        by_work: dict[ObjectRef, RevalidationWorkResultV2] = {}
        for result in work_results:
            try:
                current = RevalidationWorkResultV2.model_validate(result.model_dump(mode="python"))
                validate_revalidation_work_result_v2_identity(current)
            except (ValidationError, ValueError) as exc:
                raise UserPlanApplicationPolicyError(
                    "revalidation work result is stale or malformed"
                ) from exc
            if current.work_item_ref in by_work:
                raise UserPlanApplicationPolicyError("revalidation report contains duplicate work results")
            by_work[current.work_item_ref] = current
        expected_work_refs = tuple(revalidation_work_item_v2_ref(work) for work in current_plan.work_items)
        if not set(by_work).issubset(expected_work_refs):
            raise UserPlanApplicationPolicyError("revalidation report contains a result outside the plan")
        missing_work_refs = tuple(work_ref for work_ref in expected_work_refs if work_ref not in by_work)
        if missing_work_refs and not _missing_work_is_blocked(
            plan=current_plan,
            results_by_work=by_work,
            missing_work_refs=missing_work_refs,
        ):
            raise UserPlanApplicationPolicyError("revalidation report has unaccounted incomplete work")
        ordered_results = tuple(by_work[work_ref] for work_ref in expected_work_refs if work_ref in by_work)
        outcomes = {result.outcome for result in ordered_results}
        if RevalidationWorkOutcomeV2.TERMINAL_FAILURE in outcomes:
            outcome = DirectedRevalidationReportOutcomeV2.FAILED
        elif outcomes.intersection(
            {
                RevalidationWorkOutcomeV2.BLOCKED_POLICY,
                RevalidationWorkOutcomeV2.BLOCKED_CAPABILITY,
            }
        ):
            outcome = DirectedRevalidationReportOutcomeV2.BLOCKED
        else:
            if missing_work_refs:
                raise UserPlanApplicationPolicyError(
                    "complete revalidation report requires every work result"
                )
            outcome = DirectedRevalidationReportOutcomeV2.COMPLETE
        replacement_refs = (
            _unique_refs(
                (
                    *current_application.replacement_root_refs,
                    *(output for result in ordered_results for output in result.output_refs),
                )
            )
            if outcome is DirectedRevalidationReportOutcomeV2.COMPLETE
            else ()
        )
        if set(replacement_refs).intersection(current_application.invalidated_object_refs):
            raise UserPlanApplicationPolicyError("revalidation outputs reuse invalidated prior authority")
        return DirectedRevalidationReportV2.create(
            application_ref=application_ref,
            plan_ref=directed_revalidation_plan_v2_ref(current_plan),
            work_results=ordered_results,
            replacement_current_refs=replacement_refs,
            invalidated_prior_refs=current_application.invalidated_object_refs,
            excluded_item_ids=current_application.excluded_item_ids,
            required_checkpoint_reapprovals=(current_application.required_checkpoint_reapprovals),
            outcome=outcome,
            audit=audit,
        )

    def validate_current(
        self,
        report: DirectedRevalidationReportV2,
        *,
        application: UserPlanApplicationV2,
        plan: DirectedRevalidationPlanV2,
        work_results: tuple[RevalidationWorkResultV2, ...],
    ) -> None:
        try:
            validate_directed_revalidation_report_v2_identity(report)
        except (ValidationError, ValueError) as exc:
            raise UserPlanApplicationPolicyError(
                "directed revalidation report is stale or malformed"
            ) from exc
        rebuilt = self.compile_report(
            application=application,
            plan=plan,
            work_results=work_results,
            audit=report.audit,
        )
        if rebuilt != report:
            raise UserPlanApplicationPolicyError(
                "directed revalidation report does not match current results"
            )


def _admit_application_sources(
    *,
    decision_commit: UserDecisionCommitResultV2,
    job_spec: DatasetJobSpecV2,
    resolved_job_work_graph: ResolvedJobWorkGraphV2,
    policy: UserPlanApplicationPolicyV2,
) -> tuple[
    UserDecisionCommitResultV2,
    DatasetJobSpecV2,
    ResolvedJobWorkGraphV2,
    UserPlanApplicationPolicyV2,
]:
    try:
        commit = UserDecisionCommitResultV2.model_validate(decision_commit.model_dump(mode="python"))
        job = DatasetJobSpecV2.model_validate(job_spec.model_dump(mode="python"))
        graph = ResolvedJobWorkGraphV2.model_validate(resolved_job_work_graph.model_dump(mode="python"))
        current_policy = UserPlanApplicationPolicyV2.model_validate(policy.model_dump(mode="python"))
        validate_user_decision_commit_result_v2_identity(commit)
        validate_resolved_job_work_graph_v2_identity(graph)
        validate_user_plan_application_policy_v2_identity(current_policy)
    except (ValidationError, ValueError) as exc:
        raise UserPlanApplicationPolicyError("user plan application source is stale or malformed") from exc
    if (
        commit.outcome is not UserDecisionCommitOutcomeV2.ADJUSTMENT_ACCEPTED
        or commit.adjustment_effect is None
    ):
        raise UserPlanApplicationPolicyError("user plan application requires ADJUSTMENT_ACCEPTED")
    job_ref = dataset_job_spec_v2_ref(job)
    if (
        commit.job_id != job.job_id
        or commit.dataset_job_spec_ref != job_ref
        or graph.job_id != job.job_id
        or graph.dataset_job_spec_ref != job_ref
    ):
        raise UserPlanApplicationPolicyError("user plan application Job authority is stale")
    return commit, job, graph, current_policy


def _admit_item_sources(
    values: tuple[RevalidationItemSource, ...],
    graph: ResolvedJobWorkGraphV2,
) -> tuple[RevalidationItemSource, ...]:
    by_item: dict[str, RevalidationItemSource] = {}
    expected_trace_by_item = dict(zip(graph.item_ids, graph.source_trace_refs, strict=True))
    for value in values:
        if value.item_id in by_item:
            raise UserPlanApplicationPolicyError("revalidation Item sources must be unique")
        if (
            value.item_id not in expected_trace_by_item
            or value.source_trace_ref != expected_trace_by_item[value.item_id]
        ):
            raise UserPlanApplicationPolicyError("revalidation Item source does not match the work graph")
        if not value.authority_refs:
            raise UserPlanApplicationPolicyError("revalidation Item source requires authority refs")
        authority_refs = _unique_refs(value.authority_refs)
        if authority_refs != value.authority_refs:
            raise UserPlanApplicationPolicyError("revalidation Item authority refs must be canonical")
        stages = _admit_stage_sources(
            value.stages,
            label=f"revalidation Item {value.item_id} stages",
        )
        requirement_ids = tuple(sorted(set(value.environment_requirement_ids)))
        if requirement_ids != value.environment_requirement_ids:
            raise UserPlanApplicationPolicyError("environment requirement IDs must be canonical")
        by_item[value.item_id] = RevalidationItemSource(
            item_id=value.item_id,
            source_trace_ref=value.source_trace_ref,
            authority_refs=authority_refs,
            stages=stages,
            environment_requirement_ids=requirement_ids,
        )
    if set(by_item) != set(graph.item_ids):
        raise UserPlanApplicationPolicyError("revalidation Item sources must cover the exact work graph")
    return tuple(by_item[item_id] for item_id in sorted(by_item))


def _admit_stage_sources(
    values: tuple[RevalidationStageSource, ...],
    *,
    label: str,
) -> tuple[RevalidationStageSource, ...]:
    ordered = tuple(sorted(values, key=lambda value: list(StageNameV2).index(value.stage)))
    stages = tuple(value.stage for value in ordered)
    if len(stages) != len(set(stages)):
        raise UserPlanApplicationPolicyError(f"{label} must use unique stages")
    normalized: list[RevalidationStageSource] = []
    for value in ordered:
        outputs = _unique_refs(value.output_refs)
        if not outputs or outputs != value.output_refs:
            raise UserPlanApplicationPolicyError(f"{label} output refs must be non-empty and canonical")
        normalized.append(RevalidationStageSource(stage=value.stage, output_refs=outputs))
    return tuple(normalized)


def _admit_producer(
    *,
    commit: UserDecisionCommitResultV2,
    producer_result: AdjustmentProducerResult,
    item_sources: tuple[RevalidationItemSource, ...],
) -> _ProducerAuthority:
    effect = commit.adjustment_effect
    assert effect is not None
    if isinstance(producer_result, LabelPlanAdjustmentResultV2):
        return _admit_label_producer(effect, producer_result, item_sources)
    if isinstance(producer_result, EnvironmentStrategyAdjustmentResultV2):
        return _admit_environment_producer(
            commit,
            producer_result,
            item_sources,
        )
    if isinstance(producer_result, FinalDatasetAdjustmentResultV2):
        return _admit_final_producer(effect, producer_result, item_sources)
    return _admit_task_rewrite_producer(
        effect,
        producer_result,
        item_sources,
    )


def _admit_label_producer(
    effect: UserDecisionAdjustmentEffectV2,
    result: LabelPlanAdjustmentResultV2,
    item_sources: tuple[RevalidationItemSource, ...],
) -> _ProducerAuthority:
    try:
        validate_label_plan_adjustment_result_v2_identity(result)
    except (ValidationError, ValueError) as exc:
        raise UserPlanApplicationPolicyError("label adjustment producer is stale or malformed") from exc
    _require_effect_match(
        effect=effect,
        checkpoint=ApprovalCheckpoint.LABEL_PLAN,
        producer_ref=label_plan_adjustment_result_v2_ref(result),
        source_request_ref=result.source_request_ref,
        adjustments=result.adjustments,
        resulting_object_ref=result.replacement_label_plan_ref,
        related_result_refs=(result.replacement_label_spec_ref,),
        invalidation_scope=result.invalidation_scope,
    )
    affected = tuple(source.item_id for source in item_sources)
    return _ProducerAuthority(
        source_root_refs=_unique_refs((result.source_label_spec_ref, result.source_label_plan_ref)),
        replacement_root_refs=_unique_refs(
            (
                result.replacement_label_spec_ref,
                result.replacement_label_plan_ref,
            )
        ),
        invalidated_root_refs=result.invalidation_scope.object_refs,
        affected_item_ids=affected,
        excluded_item_ids=(),
        restart_by_item=tuple((item_id, StageNameV2.LABEL) for item_id in affected),
    )


def _admit_environment_producer(
    commit: UserDecisionCommitResultV2,
    result: EnvironmentStrategyAdjustmentResultV2,
    item_sources: tuple[RevalidationItemSource, ...],
) -> _ProducerAuthority:
    try:
        validate_environment_strategy_adjustment_result_v2_identity(result)
    except (ValidationError, ValueError) as exc:
        raise UserPlanApplicationPolicyError("environment adjustment producer is stale or malformed") from exc
    effect = commit.adjustment_effect
    assert effect is not None
    _require_effect_match(
        effect=effect,
        checkpoint=ApprovalCheckpoint.ENVIRONMENT_STRATEGY,
        producer_ref=environment_strategy_adjustment_result_v2_ref(result),
        source_request_ref=result.source_request_ref,
        adjustments=result.adjustments,
        resulting_object_ref=result.replacement_strategy_ref,
        related_result_refs=(),
        invalidation_scope=result.invalidation_scope,
    )
    decisions = {
        value.requirement_id: value.strategy for value in commit.decision_record.environment_decisions
    }
    affected_subjects = set(effect.source_subject_refs)
    restart: list[tuple[str, StageNameV2]] = []
    excluded: list[str] = []
    for source in item_sources:
        if not affected_subjects.intersection(source.authority_refs):
            continue
        requirement_ids = (
            source.environment_requirement_ids if source.environment_requirement_ids else tuple(decisions)
        )
        try:
            strategies = {decisions[value] for value in requirement_ids}
        except KeyError as exc:
            raise UserPlanApplicationPolicyError(
                "Item environment requirements do not match the decision"
            ) from exc
        if EnvironmentStrategyChoice.EXCLUDE_TASK in strategies:
            excluded.append(source.item_id)
            restart.append((source.item_id, StageNameV2.ITEM_QUALITY))
        elif EnvironmentStrategyChoice.REWRITE_STANDARD_ENV in strategies:
            restart.append((source.item_id, StageNameV2.TASK_AUTHORING))
        else:
            restart.append((source.item_id, StageNameV2.ATTACHMENT))
    if not restart:
        raise UserPlanApplicationPolicyError("environment adjustment affects no current Item")
    affected = tuple(item_id for item_id, _ in restart)
    return _ProducerAuthority(
        source_root_refs=(result.source_strategy_ref,),
        replacement_root_refs=(result.replacement_strategy_ref,),
        invalidated_root_refs=result.invalidation_scope.object_refs,
        affected_item_ids=affected,
        excluded_item_ids=tuple(sorted(excluded)),
        restart_by_item=tuple(sorted(restart)),
    )


def _admit_final_producer(
    effect: UserDecisionAdjustmentEffectV2,
    result: FinalDatasetAdjustmentResultV2,
    item_sources: tuple[RevalidationItemSource, ...],
) -> _ProducerAuthority:
    try:
        validate_final_dataset_adjustment_result_v2_identity(result)
    except (ValidationError, ValueError) as exc:
        raise UserPlanApplicationPolicyError("final adjustment producer is stale or malformed") from exc
    _require_effect_match(
        effect=effect,
        checkpoint=ApprovalCheckpoint.FINAL_DATASET_REVIEW,
        producer_ref=final_dataset_adjustment_result_v2_ref(result),
        source_request_ref=result.source_request_ref,
        adjustments=result.adjustments,
        resulting_object_ref=result.replacement_preview_ref,
        related_result_refs=(),
        invalidation_scope=result.invalidation_scope,
    )
    directives = {
        value.target_path.removeprefix("items[").split("]", 1)[0]: value.value for value in result.adjustments
    }
    target_ids = set(result.target_item_ids)
    restart: list[tuple[str, StageNameV2]] = []
    excluded: list[str] = []
    for source in item_sources:
        candidate_ids = (
            source.item_id,
            *(ref.object_id for ref in source.authority_refs),
        )
        matched_id = next(
            (value for value in candidate_ids if value in target_ids),
            None,
        )
        if matched_id is None:
            continue
        selector = hashlib.sha256(matched_id.encode()).hexdigest()
        directive = directives.get(selector)
        if directive == "REBUILD_FROM_TASK_AUTHORING":
            stage = StageNameV2.TASK_AUTHORING
        elif directive == "REBUILD_FROM_ATTACHMENT":
            stage = StageNameV2.ATTACHMENT
        elif directive == "REBUILD_FROM_ITEM_QUALITY":
            stage = StageNameV2.ITEM_QUALITY
        elif directive == "EXCLUDE_ITEM":
            stage = StageNameV2.ITEM_QUALITY
            excluded.append(source.item_id)
        else:
            raise UserPlanApplicationPolicyError(
                "final adjustment target does not have one restart directive"
            )
        restart.append((source.item_id, stage))
    if len(restart) != len(target_ids):
        raise UserPlanApplicationPolicyError("final adjustment targets do not match current Items")
    affected = tuple(item_id for item_id, _ in restart)
    return _ProducerAuthority(
        source_root_refs=(result.source_preview_ref,),
        replacement_root_refs=(result.replacement_preview_ref,),
        invalidated_root_refs=result.invalidation_scope.object_refs,
        affected_item_ids=affected,
        excluded_item_ids=tuple(sorted(excluded)),
        restart_by_item=tuple(sorted(restart)),
    )


def _admit_task_rewrite_producer(
    effect: UserDecisionAdjustmentEffectV2,
    source: TaskRewriteApplicationSource,
    item_sources: tuple[RevalidationItemSource, ...],
) -> _ProducerAuthority:
    adjustment = TaskRewriteAdjustmentResult.model_validate(
        source.adjustment_result.model_dump(mode="python")
    )
    application_result = TaskRewriteApplicationResult.model_validate(
        source.application_result.model_dump(mode="python")
    )
    if (
        adjustment.outcome is not TaskRewritePlanCompileOutcome.PREVIEWED
        or adjustment.replacement_plan_version is None
        or adjustment.replacement_preview_safety_gate is None
        or adjustment.replacement_preview is None
        or adjustment.invalidation is None
    ):
        raise UserPlanApplicationPolicyError("task rewrite adjustment requires complete PREVIEWED output")
    source_plan = adjustment.source_plan_version
    replacement_plan = adjustment.replacement_plan_version
    gate = adjustment.replacement_preview_safety_gate
    preview = adjustment.replacement_preview
    invalidation = adjustment.invalidation
    if adjustment.result_id != f"task-rewrite-adjustment-result://sha256/{adjustment.result_sha256}":
        raise UserPlanApplicationPolicyError("task rewrite adjustment result identity is stale")
    if source_plan.task_rewrite_plan_version_sha256 != task_rewrite_plan_version_carried_sha256(
        source_plan
    ) or source_plan.task_rewrite_plan_version_id != (
        f"task-rewrite-plan-version://sha256/{source_plan.task_rewrite_plan_version_sha256}"
    ):
        raise UserPlanApplicationPolicyError("task rewrite source plan version is stale")
    if replacement_plan.task_rewrite_plan_version_sha256 != task_rewrite_plan_version_carried_sha256(
        replacement_plan
    ) or replacement_plan.task_rewrite_plan_version_id != (
        f"task-rewrite-plan-version://sha256/{replacement_plan.task_rewrite_plan_version_sha256}"
    ):
        raise UserPlanApplicationPolicyError("task rewrite replacement plan version is stale")
    if (
        gate.gate_sha256 != task_rewrite_preview_safety_gate_carried_sha256(gate)
        or gate.gate_id != f"task-rewrite-preview-safety-gate://sha256/{gate.gate_sha256}"
    ):
        raise UserPlanApplicationPolicyError("task rewrite replacement preview gate is stale")
    if (
        preview.preview_sha256 != task_rewrite_plan_preview_carried_sha256(preview)
        or preview.preview_id != f"task-rewrite-plan-preview://sha256/{preview.preview_sha256}"
    ):
        raise UserPlanApplicationPolicyError("task rewrite replacement preview is stale")
    if (
        invalidation.invalidation_sha256 != task_contract_invalidation_carried_sha256(invalidation)
        or invalidation.invalidation_id
        != f"task-contract-invalidation://sha256/{invalidation.invalidation_sha256}"
    ):
        raise UserPlanApplicationPolicyError("task rewrite invalidation is stale")
    producer_ref = ObjectRef(
        object_type="task-rewrite-adjustment-result",
        object_id=adjustment.result_id,
        object_version=adjustment.policy_version,
        object_sha256=adjustment.result_sha256,
    )
    source_plan_ref = task_rewrite_plan_version_ref(source_plan)
    replacement_plan_ref = task_rewrite_plan_version_ref(replacement_plan)
    invalidation_ref = task_contract_invalidation_ref(invalidation)
    gate_ref = task_rewrite_preview_safety_gate_ref(gate)
    preview_ref = task_rewrite_plan_preview_ref(preview)
    if (
        effect.source_plan_ref != source_plan_ref
        or preview.source_plan_version_ref != replacement_plan_ref
        or preview.safety_gate_ref != gate_ref
        or invalidation.source_plan_version_ref != source_plan_ref
        or invalidation.replacement_plan_version_ref != replacement_plan_ref
    ):
        raise UserPlanApplicationPolicyError("task rewrite adjustment output binding is stale")
    invalidation_scope = InvalidationScope(
        object_refs=_unique_refs(invalidation.invalidated_object_refs),
        stages=tuple(sorted(stage.value for stage in invalidation.required_rebuild_stages)),
    )
    _require_effect_match(
        effect=effect,
        checkpoint=ApprovalCheckpoint.TASK_REWRITE_PLAN,
        producer_ref=producer_ref,
        source_request_ref=effect.source_request_ref,
        adjustments=effect.adjustments,
        resulting_object_ref=replacement_plan_ref,
        related_result_refs=(gate_ref, preview_ref, invalidation_ref),
        invalidation_scope=invalidation_scope,
    )
    replacement_contract_set = application_result.replacement_contract_set
    if replacement_contract_set.contract_set_sha256 != r4_task_contract_set_carried_sha256(
        replacement_contract_set
    ) or replacement_contract_set.contract_set_id != (
        f"r4-task-contract-set://sha256/{replacement_contract_set.contract_set_sha256}"
    ):
        raise UserPlanApplicationPolicyError("task rewrite replacement contract set is stale")
    application = application_result.application
    if (
        application.application_sha256 != task_rewrite_application_carried_sha256(application)
        or application.application_id
        != (f"task-rewrite-application://sha256/{application.application_sha256}")
        or application.source_plan_version_ref != source_plan_ref
        or application.replacement_plan_version_ref != replacement_plan_ref
        or application.invalidation_ref != invalidation_ref
        or application.source_contract_set_ref != invalidation.source_contract_set_ref
        or application.replacement_contract_set_ref != r4_task_contract_set_ref(replacement_contract_set)
    ):
        raise UserPlanApplicationPolicyError("task rewrite application binding is stale")
    expected_result_sha = rewrite_payload_sha256(
        {
            "application_ref": _ref_payload(task_rewrite_application_ref(application)),
            "replacement_contract_set_ref": _ref_payload(r4_task_contract_set_ref(replacement_contract_set)),
            "policy_version": application_result.policy_version,
        }
    )
    if (
        application_result.result_sha256 != expected_result_sha
        or application_result.result_id != f"task-rewrite-application-result://sha256/{expected_result_sha}"
    ):
        raise UserPlanApplicationPolicyError("task rewrite application result is stale")
    trigger_refs = {
        *effect.source_subject_refs,
        invalidation.source_contract_set_ref,
        *invalidation.invalidated_object_refs,
    }
    affected = tuple(
        source.item_id for source in item_sources if trigger_refs.intersection(source.authority_refs)
    )
    if not affected:
        raise UserPlanApplicationPolicyError("task rewrite adjustment affects no current Item")
    return _ProducerAuthority(
        source_root_refs=_unique_refs(
            (
                source_plan_ref,
                invalidation.source_contract_set_ref,
            )
        ),
        replacement_root_refs=_unique_refs(
            (
                replacement_plan_ref,
                r4_task_contract_set_ref(replacement_contract_set),
            )
        ),
        invalidated_root_refs=invalidation.invalidated_object_refs,
        affected_item_ids=affected,
        excluded_item_ids=(),
        restart_by_item=tuple((item_id, StageNameV2.TASK_AUTHORING) for item_id in affected),
    )


def _require_effect_match(
    *,
    effect: UserDecisionAdjustmentEffectV2,
    checkpoint: ApprovalCheckpoint,
    producer_ref: ObjectRef,
    source_request_ref: ObjectRef,
    adjustments: tuple[TypedAdjustment, ...],
    resulting_object_ref: ObjectRef,
    related_result_refs: tuple[ObjectRef, ...],
    invalidation_scope: InvalidationScope,
) -> None:
    if (
        effect.checkpoint is not checkpoint
        or effect.producer_result_ref != producer_ref
        or effect.source_request_ref != source_request_ref
        or effect.adjustments != adjustments
        or effect.resulting_object_ref != resulting_object_ref
        or set(effect.related_result_refs) != set(related_result_refs)
        or effect.invalidation_scope != invalidation_scope
    ):
        raise UserPlanApplicationPolicyError("adjustment producer does not match the committed effect")


def _application_restart_stages(
    restart_by_item: dict[str, StageNameV2],
) -> tuple[StageNameV2, ...]:
    stages = {stage for restart in restart_by_item.values() for stage in _stage_closure(restart)}
    return tuple(stage for stage in StageNameV2 if stage in stages)


def _stage_closure(start: StageNameV2) -> tuple[StageNameV2, ...]:
    try:
        position = _EXECUTION_STAGES.index(start)
    except ValueError as exc:
        raise UserPlanApplicationPolicyError("revalidation restart stage is not executable") from exc
    return _EXECUTION_STAGES[position:]


def _invalidated_refs(
    *,
    commit: UserDecisionCommitResultV2,
    producer_result: AdjustmentProducerResult,
    authority: _ProducerAuthority,
    item_sources: tuple[RevalidationItemSource, ...],
    job_stage_sources: tuple[RevalidationStageSource, ...],
    restart_by_item: dict[str, StageNameV2],
) -> tuple[ObjectRef, ...]:
    refs: list[ObjectRef] = list(authority.invalidated_root_refs)
    source_by_item = {source.item_id: source for source in item_sources}
    for item_id, restart in restart_by_item.items():
        closure = set(_stage_closure(restart))
        refs.extend(
            output
            for stage in source_by_item[item_id].stages
            if stage.stage in closure
            for output in stage.output_refs
        )
    restarted = set(_application_restart_stages(restart_by_item))
    refs.extend(
        output for stage in job_stage_sources if stage.stage in restarted for output in stage.output_refs
    )
    if isinstance(producer_result, FinalDatasetAdjustmentResultV2):
        refs.extend(
            (
                producer_result.source_preview_ref,
                commit.request_ref,
                commit.decision_record_ref,
            )
        )
    return _unique_refs(refs)


def _preserved_refs(
    *,
    authority: _ProducerAuthority,
    item_sources: tuple[RevalidationItemSource, ...],
    job_stage_sources: tuple[RevalidationStageSource, ...],
    restart_by_item: dict[str, StageNameV2],
    invalidated_refs: tuple[ObjectRef, ...],
) -> tuple[ObjectRef, ...]:
    invalidated = set(invalidated_refs)
    replacements = set(authority.replacement_root_refs)
    refs: list[ObjectRef] = []
    for source in item_sources:
        closure = (
            set(_stage_closure(restart_by_item[source.item_id]))
            if source.item_id in restart_by_item
            else set()
        )
        refs.extend((source.source_trace_ref, *source.authority_refs))
        refs.extend(
            output for stage in source.stages if stage.stage not in closure for output in stage.output_refs
        )
    restarted = set(_application_restart_stages(restart_by_item))
    refs.extend(
        output for stage in job_stage_sources if stage.stage not in restarted for output in stage.output_refs
    )
    return _unique_refs(ref for ref in refs if ref not in invalidated and ref not in replacements)


def _compile_work_items(
    *,
    application: UserPlanApplicationV2,
    authority: _ProducerAuthority,
    item_sources: tuple[RevalidationItemSource, ...],
    job_stage_sources: tuple[RevalidationStageSource, ...],
    restart_by_item: dict[str, StageNameV2],
) -> tuple[RevalidationWorkItemV2, ...]:
    application_ref = user_plan_application_v2_ref(application)
    excluded = set(authority.excluded_item_ids)
    source_by_item = {source.item_id: source for source in item_sources}
    works: list[RevalidationWorkItemV2] = []
    latest_by_item: dict[str, RevalidationWorkItemV2] = {}
    for stage in _ITEM_EXECUTION_STAGES:
        for item_id in sorted(restart_by_item):
            if item_id in excluded or stage not in _stage_closure(restart_by_item[item_id]):
                continue
            source = source_by_item[item_id]
            previous = latest_by_item.get(item_id)
            invalidated_inputs = _work_invalidated_inputs(
                application=application,
                source=source,
                stage=stage,
            )
            replacement_inputs = _unique_refs(
                (
                    source.source_trace_ref,
                    *(authority.replacement_root_refs if previous is None else ()),
                )
            )
            work = RevalidationWorkItemV2.create(
                application_ref=application_ref,
                scope=RevalidationWorkScopeV2.ITEM,
                item_id=item_id,
                stage=stage,
                invalidated_input_refs=invalidated_inputs,
                replacement_input_refs=tuple(
                    ref for ref in replacement_inputs if ref not in set(invalidated_inputs)
                ),
                depends_on_work_item_refs=(
                    (revalidation_work_item_v2_ref(previous),) if previous is not None else ()
                ),
                required_output_types=_OUTPUT_TYPES[stage],
                hard_gate_required=stage in _HARD_GATE_STAGES,
            )
            works.append(work)
            latest_by_item[item_id] = work
    batch_sources = tuple(
        output
        for value in job_stage_sources
        if value.stage is StageNameV2.BATCH_QUALITY
        for output in value.output_refs
    )
    if not batch_sources:
        batch_sources = tuple(
            output
            for item_id in restart_by_item
            for stage in source_by_item[item_id].stages
            if stage.stage is StageNameV2.ITEM_QUALITY
            for output in stage.output_refs
        )
    batch_invalidated = _unique_refs(batch_sources or application.invalidated_object_refs)
    preserved_item_quality = tuple(
        output
        for source in item_sources
        if source.item_id not in restart_by_item
        for stage in source.stages
        if stage.stage is StageNameV2.ITEM_QUALITY
        for output in stage.output_refs
    )
    batch = RevalidationWorkItemV2.create(
        application_ref=application_ref,
        scope=RevalidationWorkScopeV2.JOB,
        item_id=None,
        stage=StageNameV2.BATCH_QUALITY,
        invalidated_input_refs=batch_invalidated,
        replacement_input_refs=_unique_refs((*authority.replacement_root_refs, *preserved_item_quality)),
        depends_on_work_item_refs=_unique_refs(
            revalidation_work_item_v2_ref(work) for work in latest_by_item.values()
        ),
        required_output_types=_OUTPUT_TYPES[StageNameV2.BATCH_QUALITY],
        hard_gate_required=True,
    )
    works.append(batch)
    return tuple(works)


def _work_invalidated_inputs(
    *,
    application: UserPlanApplicationV2,
    source: RevalidationItemSource,
    stage: StageNameV2,
) -> tuple[ObjectRef, ...]:
    exact = tuple(output for value in source.stages if value.stage is stage for output in value.output_refs)
    if exact:
        return exact
    earlier = tuple(
        output
        for value in reversed(source.stages)
        if list(StageNameV2).index(value.stage) < list(StageNameV2).index(stage)
        for output in value.output_refs
    )
    return _unique_refs(earlier or application.invalidated_object_refs)


def _preflight_application_budgets(
    *,
    policy: UserPlanApplicationPolicyV2,
    adjustment_count: int,
    invalidated_refs: tuple[ObjectRef, ...],
    preserved_refs: tuple[ObjectRef, ...],
    affected_item_ids: tuple[str, ...],
) -> None:
    checks = (
        (
            adjustment_count,
            policy.max_adjustments,
            "adjustment limit",
        ),
        (
            len(invalidated_refs),
            policy.max_invalidated_refs,
            "invalidated ref limit",
        ),
        (
            len(preserved_refs),
            policy.max_preserved_refs,
            "preserved ref limit",
        ),
        (
            len(affected_item_ids),
            policy.max_affected_items,
            "affected Item limit",
        ),
    )
    for observed, maximum, label in checks:
        if observed > maximum:
            raise UserPlanApplicationPolicyError(f"user plan application exceeds the {label}")


def _missing_work_is_blocked(
    *,
    plan: DirectedRevalidationPlanV2,
    results_by_work: dict[ObjectRef, RevalidationWorkResultV2],
    missing_work_refs: tuple[ObjectRef, ...],
) -> bool:
    work_by_ref = {revalidation_work_item_v2_ref(work): work for work in plan.work_items}
    unsuccessful = {
        work_ref
        for work_ref, result in results_by_work.items()
        if result.outcome is not RevalidationWorkOutcomeV2.SUCCEEDED
    }
    if not unsuccessful:
        return False

    def has_unsuccessful_ancestor(
        work_ref: ObjectRef,
        visited: frozenset[ObjectRef],
    ) -> bool:
        if work_ref in unsuccessful:
            return True
        if work_ref in visited:
            return False
        return any(
            has_unsuccessful_ancestor(
                dependency,
                visited | {work_ref},
            )
            for dependency in work_by_ref[work_ref].depends_on_work_item_refs
        )

    return all(has_unsuccessful_ancestor(work_ref, frozenset()) for work_ref in missing_work_refs)


def _validate_dependency_results(
    *,
    work: RevalidationWorkItemV2,
    dependency_results: tuple[RevalidationWorkResultV2, ...],
) -> None:
    by_work: dict[ObjectRef, RevalidationWorkResultV2] = {}
    for result in dependency_results:
        try:
            current = RevalidationWorkResultV2.model_validate(result.model_dump(mode="python"))
            validate_revalidation_work_result_v2_identity(current)
        except (ValidationError, ValueError) as exc:
            raise UserPlanApplicationPolicyError(
                "revalidation dependency result is stale or malformed"
            ) from exc
        if current.work_item_ref in by_work:
            raise UserPlanApplicationPolicyError("revalidation dependency results must be unique")
        by_work[current.work_item_ref] = current
    if set(by_work) != set(work.depends_on_work_item_refs):
        raise UserPlanApplicationPolicyError("revalidation dependency results are incomplete")
    if any(result.outcome is not RevalidationWorkOutcomeV2.SUCCEEDED for result in by_work.values()):
        raise UserPlanApplicationPolicyError("revalidation dependency did not succeed")


def _result_outcome(
    *,
    work: RevalidationWorkItemV2,
    stage_result: StageResultRecord,
    policy: UserPlanApplicationPolicyV2,
    invalidated_refs: tuple[ObjectRef, ...],
) -> tuple[
    RevalidationWorkOutcomeV2,
    tuple[ObjectRef, ...],
    str | None,
]:
    if stage_result.status is StageRunStatus.SUCCEEDED:
        outputs = _unique_refs(stage_result.output_refs)
        if (
            not outputs
            or len(outputs) != len(stage_result.output_refs)
            or len(outputs) > policy.max_outputs_per_work
            or {ref.object_type for ref in outputs} != set(work.required_output_types)
            or set(outputs).intersection(invalidated_refs)
        ):
            raise UserPlanApplicationPolicyError(
                "successful revalidation output does not match the work contract"
            )
        return RevalidationWorkOutcomeV2.SUCCEEDED, outputs, None
    outcome_by_status = {
        StageRunStatus.BLOCKED_POLICY: (RevalidationWorkOutcomeV2.BLOCKED_POLICY),
        StageRunStatus.BLOCKED_CAPABILITY: (RevalidationWorkOutcomeV2.BLOCKED_CAPABILITY),
        StageRunStatus.TERMINAL_FAILURE: (RevalidationWorkOutcomeV2.TERMINAL_FAILURE),
    }
    outcome = outcome_by_status.get(stage_result.status)
    if outcome is None:
        raise UserPlanApplicationPolicyError("revalidation result is not terminal for this application")
    if stage_result.output_refs or stage_result.failure is None:
        raise UserPlanApplicationPolicyError("non-success revalidation cannot carry outputs")
    failure_code = stage_result.failure.code
    if len(failure_code) > policy.max_failure_code_characters:
        raise UserPlanApplicationPolicyError("revalidation failure code exceeds the policy limit")
    return outcome, (), failure_code


def _unique_refs(values: Iterable[ObjectRef]) -> tuple[ObjectRef, ...]:
    return tuple(
        sorted(
            set(values),
            key=lambda ref: (
                ref.object_type,
                ref.object_id,
                ref.object_version,
                ref.object_sha256,
            ),
        )
    )


def _ref_payload(value: ObjectRef) -> dict[str, object]:
    return value.model_dump(mode="json", exclude_none=False)
