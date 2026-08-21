from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pydantic import ValidationError

from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    UserApprovalPolicy,
)
from eval_factory.contracts.approval_application_v2 import (
    DirectedRevalidationReportOutcomeV2,
    DirectedRevalidationReportV2,
    directed_revalidation_report_v2_ref,
    validate_directed_revalidation_report_v2_identity,
)
from eval_factory.contracts.approval_decision_v2 import (
    UserDecisionCommitOutcomeV2,
    UserDecisionCommitResultV2,
    user_decision_commit_result_v2_ref,
)
from eval_factory.contracts.approval_v2 import (
    user_approval_policy_ref,
)
from eval_factory.contracts.attachment_v2 import (
    AttachmentReconstructionResultV2,
    attachment_reconstruction_result_v2_ref,
    validate_attachment_reconstruction_result_v2_identity,
)
from eval_factory.contracts.batch_quality_v2 import (
    BatchQualityReportV2,
    batch_quality_report_v2_ref,
    validate_batch_quality_report_v2_identity,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
    LabelDecisionValueV2,
    LabelExecutionStatus,
    label_decision_ref,
)
from eval_factory.contracts.orchestration import ItemStatus
from eval_factory.contracts.orchestration_v2 import (
    DatasetJobSpecV2,
    ResolvedJobWorkGraphV2,
    StageNameV2,
    resolved_job_work_graph_v2_ref,
)
from eval_factory.contracts.quality_v2 import (
    ItemQualityCompilationResultV2,
    environment_spec_v2_ref,
    final_package_manifest_ref,
    item_quality_compilation_result_ref,
    provenance_manifest_v2_ref,
    quality_report_v2_ref,
)
from eval_factory.contracts.release import (
    EvaluationItemComponents,
    ReleaseChannel,
)
from eval_factory.contracts.release_projection_v2 import (
    EvaluationItemReleaseSubjectV2,
    ItemReleaseProjectionV2,
    ReleaseProjectionPhaseV2,
    ReleaseProjectionPolicyV2,
    ReleaseProjectionResultV2,
    evaluation_item_v2_carried_sha256,
    item_release_projection_v2_ref,
    query_spec_carried_sha256,
    query_spec_ref,
    release_decision_v2_carried_sha256,
    release_projection_result_v2_ref,
    validate_release_projection_policy_v2_identity,
    validate_release_projection_result_v2_identity,
)
from eval_factory.contracts.release_v2 import (
    CheckpointDecisionBinding,
    EvaluationItemV2,
    ReleaseActionV2,
    ReleaseDecisionV2,
    ReleaseStateV2,
)
from eval_factory.contracts.task import QuerySpec
from eval_factory.contracts.task_v2 import (
    R4TaskContractSetV2,
    TaskDraftPromptSafetyStatusV2,
    TaskDraftV2,
    TaskPromptSafetyGateStatusV2,
    TaskPromptSafetyGateV2,
    r4_task_contract_set_carried_sha256,
    r4_task_contract_set_ref,
    task_draft_carried_sha256,
    task_draft_ref,
    task_prompt_safety_gate_ref,
)
from eval_factory.orchestration.models import ItemRecord


class ReleaseProjectionPolicyError(ValueError):
    pass


class ReleaseProjectionPendingError(ReleaseProjectionPolicyError):
    pass


@dataclass(frozen=True, slots=True)
class EvaluationItemReleaseSource:
    job_spec: DatasetJobSpecV2
    item: ItemRecord
    resolved_job_work_graph: ResolvedJobWorkGraphV2
    source_trace_refs: tuple[ObjectRef, ...]
    label_decisions: tuple[LabelDecisionV2, ...]
    task_draft: TaskDraftV2
    task_prompt_safety_gate: TaskPromptSafetyGateV2
    task_contract_set: R4TaskContractSetV2
    attachment_result: AttachmentReconstructionResultV2
    item_quality: ItemQualityCompilationResultV2
    batch_quality: BatchQualityReportV2
    approval_policy: UserApprovalPolicy
    decision_commits: tuple[UserDecisionCommitResultV2, ...]
    relevant_revalidation_application_refs: tuple[ObjectRef, ...]
    revalidation_reports: tuple[DirectedRevalidationReportV2, ...]
    current_head_refs: tuple[ObjectRef, ...]
    not_required_checkpoints: tuple[ApprovalCheckpoint, ...] = ()


@dataclass(frozen=True, slots=True)
class ReleaseProjectionCompilation:
    result: ReleaseProjectionResultV2


@dataclass(frozen=True, slots=True)
class _PreparedReleaseSource:
    source: EvaluationItemReleaseSource
    query_spec: QuerySpec
    components: EvaluationItemComponents
    required_checkpoints: tuple[ApprovalCheckpoint, ...]
    not_required_checkpoints: tuple[ApprovalCheckpoint, ...]
    plan_bindings: tuple[CheckpointDecisionBinding, ...]
    current_decision_refs: tuple[ObjectRef, ...]
    relevant_application_refs: tuple[ObjectRef, ...]
    report_refs: tuple[ObjectRef, ...]
    current_head_refs: tuple[ObjectRef, ...]


class ReleaseProjectionCompiler:
    def compile_candidate(
        self,
        *,
        source: EvaluationItemReleaseSource,
        policy: ReleaseProjectionPolicyV2,
        previous_result: ReleaseProjectionResultV2 | None,
        audit: ContractAudit,
    ) -> ReleaseProjectionCompilation:
        current_policy = _validate_policy(policy)
        prepared = _prepare_source(source, current_policy)
        previous = _validate_previous(previous_result, current_policy)
        revision = 1 if previous is None else previous.item_projection.projection_revision + 1
        if revision > current_policy.max_chain_depth:
            raise ReleaseProjectionPolicyError("release projection chain depth exceeds policy")
        pending = tuple(
            checkpoint
            for checkpoint in prepared.required_checkpoints
            if checkpoint not in {binding.checkpoint for binding in prepared.plan_bindings}
        )
        if pending not in {(), (ApprovalCheckpoint.FINAL_DATASET_REVIEW,)}:
            raise ReleaseProjectionPolicyError("candidate may be pending only Final Dataset Review")
        subject = _release_subject(
            prepared,
            checkpoint_bindings=prepared.plan_bindings,
            audit=audit,
            policy=current_policy,
        )
        chain_id = (
            previous.release_decision.chain_id
            if previous is not None
            else _chain_id(subject.job_id, subject.item_id)
        )
        decision = _release_decision(
            subject=subject,
            action=ReleaseActionV2.REQUEST_RELEASE,
            chain_id=chain_id,
            revision=revision,
            previous_decision_ref=(previous.release_decision_ref if previous is not None else None),
            checkpoint_bindings=prepared.plan_bindings,
            audit=audit,
        )
        item = _evaluation_item(
            subject=subject,
            decision=decision,
            revision=revision,
            audit=audit,
        )
        projection = ItemReleaseProjectionV2.create(
            job_id=subject.job_id,
            item_id=subject.item_id,
            projection_revision=revision,
            chain_id=chain_id,
            previous_projection_ref=(
                item_release_projection_v2_ref(previous.item_projection) if previous is not None else None
            ),
            release_subject_ref=subject.to_ref(),
            evaluation_item_ref=_evaluation_item_ref(item),
            release_decision_ref=_release_decision_ref(decision),
            release_state=ReleaseStateV2.CANDIDATE,
            pending_checkpoints=pending,
            audit=audit,
        )
        result = ReleaseProjectionResultV2.create(
            phase=ReleaseProjectionPhaseV2.CANDIDATE,
            release_subject=subject,
            query_spec=prepared.query_spec,
            release_decision=decision,
            evaluation_item=item,
            item_projection=projection,
            previous_result_ref=(
                release_projection_result_v2_ref(previous) if previous is not None else None
            ),
            policy_ref=current_policy.to_ref(),
            audit=audit,
        )
        return ReleaseProjectionCompilation(result=result)

    def finalize(
        self,
        *,
        candidate: ReleaseProjectionResultV2,
        source: EvaluationItemReleaseSource,
        policy: ReleaseProjectionPolicyV2,
        audit: ContractAudit,
    ) -> ReleaseProjectionCompilation:
        current_policy = _validate_policy(policy)
        try:
            current_candidate = ReleaseProjectionResultV2.model_validate(candidate.model_dump(mode="python"))
            validate_release_projection_result_v2_identity(current_candidate)
        except (ValidationError, ValueError) as exc:
            raise ReleaseProjectionPolicyError("candidate release projection is stale") from exc
        if current_candidate.phase is not ReleaseProjectionPhaseV2.CANDIDATE:
            raise ReleaseProjectionPolicyError("release finalization requires a candidate result")
        if current_candidate.policy_ref != current_policy.to_ref():
            raise ReleaseProjectionPolicyError("candidate release projection policy is stale")
        prepared = _prepare_source(
            source,
            current_policy,
            allow_final_decision=True,
        )
        _require_same_candidate_authority(current_candidate, prepared)
        action, bindings = _terminal_action(
            prepared,
            candidate_item_ref=current_candidate.evaluation_item_ref,
        )
        revision = current_candidate.item_projection.projection_revision + 1
        if revision > current_policy.max_chain_depth:
            raise ReleaseProjectionPolicyError("release projection chain depth exceeds policy")
        subject = current_candidate.release_subject
        decision = _release_decision(
            subject=subject,
            action=action,
            chain_id=current_candidate.release_decision.chain_id,
            revision=revision,
            previous_decision_ref=current_candidate.release_decision_ref,
            checkpoint_bindings=bindings,
            audit=audit,
        )
        item = _evaluation_item(
            subject=subject,
            decision=decision,
            revision=revision,
            audit=audit,
            user_decision_record_refs=prepared.current_decision_refs,
        )
        projection = ItemReleaseProjectionV2.create(
            job_id=subject.job_id,
            item_id=subject.item_id,
            projection_revision=revision,
            chain_id=decision.chain_id,
            previous_projection_ref=current_candidate.item_projection_ref,
            release_subject_ref=subject.to_ref(),
            evaluation_item_ref=_evaluation_item_ref(item),
            release_decision_ref=_release_decision_ref(decision),
            release_state=decision.state,
            pending_checkpoints=(),
            audit=audit,
        )
        result = ReleaseProjectionResultV2.create(
            phase=ReleaseProjectionPhaseV2.TERMINAL,
            release_subject=subject,
            query_spec=prepared.query_spec,
            release_decision=decision,
            evaluation_item=item,
            item_projection=projection,
            previous_result_ref=current_candidate.to_ref(),
            policy_ref=current_policy.to_ref(),
            audit=audit,
        )
        return ReleaseProjectionCompilation(result=result)

    def validate_current(
        self,
        result: ReleaseProjectionResultV2,
        *,
        source: EvaluationItemReleaseSource,
        policy: ReleaseProjectionPolicyV2,
        previous_result: ReleaseProjectionResultV2 | None,
        audit: ContractAudit,
    ) -> None:
        if result.phase is ReleaseProjectionPhaseV2.CANDIDATE:
            rebuilt = self.compile_candidate(
                source=source,
                policy=policy,
                previous_result=previous_result,
                audit=audit,
            ).result
        else:
            if previous_result is None:
                raise ReleaseProjectionPolicyError("terminal currentness requires its candidate result")
            rebuilt = self.finalize(
                candidate=previous_result,
                source=source,
                policy=policy,
                audit=audit,
            ).result
        if rebuilt.to_ref() != result.to_ref():
            raise ReleaseProjectionPolicyError("release projection differs from current authority")


def _validate_policy(
    policy: ReleaseProjectionPolicyV2,
) -> ReleaseProjectionPolicyV2:
    try:
        current = ReleaseProjectionPolicyV2.model_validate(policy.model_dump(mode="python"))
        validate_release_projection_policy_v2_identity(current)
    except (ValidationError, ValueError) as exc:
        raise ReleaseProjectionPolicyError("release projection policy is stale") from exc
    return current


def _validate_previous(
    value: ReleaseProjectionResultV2 | None,
    policy: ReleaseProjectionPolicyV2,
) -> ReleaseProjectionResultV2 | None:
    if value is None:
        return None
    try:
        current = ReleaseProjectionResultV2.model_validate(value.model_dump(mode="python"))
        validate_release_projection_result_v2_identity(current)
    except (ValidationError, ValueError) as exc:
        raise ReleaseProjectionPolicyError("previous release projection is stale") from exc
    if current.policy_ref != policy.to_ref():
        raise ReleaseProjectionPolicyError("previous release projection policy is stale")
    return current


def _prepare_source(
    source: EvaluationItemReleaseSource,
    policy: ReleaseProjectionPolicyV2,
    *,
    allow_final_decision: bool = False,
) -> _PreparedReleaseSource:
    try:
        job_spec = DatasetJobSpecV2.model_validate(source.job_spec.model_dump(mode="python"))
        item = ItemRecord.model_validate(source.item.model_dump(mode="python"))
        graph = ResolvedJobWorkGraphV2.model_validate(
            source.resolved_job_work_graph.model_dump(mode="python")
        )
        resolved_job_work_graph_v2_ref(graph)
        draft = TaskDraftV2.model_validate(source.task_draft.model_dump(mode="python"))
        gate = TaskPromptSafetyGateV2.model_validate(source.task_prompt_safety_gate.model_dump(mode="python"))
        contract_set = R4TaskContractSetV2.model_validate(source.task_contract_set.model_dump(mode="python"))
        attachment = AttachmentReconstructionResultV2.model_validate(
            source.attachment_result.model_dump(mode="python")
        )
        validate_attachment_reconstruction_result_v2_identity(attachment)
        item_quality = ItemQualityCompilationResultV2.model_validate(
            source.item_quality.model_dump(mode="python")
        )
        batch_quality = BatchQualityReportV2.model_validate(source.batch_quality.model_dump(mode="python"))
        validate_batch_quality_report_v2_identity(batch_quality)
        approval = UserApprovalPolicy.model_validate(source.approval_policy.model_dump(mode="python"))
        approval_ref = user_approval_policy_ref(approval)
        decisions = tuple(
            UserDecisionCommitResultV2.model_validate(value.model_dump(mode="python"))
            for value in source.decision_commits
        )
    except (ValidationError, ValueError) as exc:
        raise ReleaseProjectionPolicyError("release source authority is stale or malformed") from exc

    if (
        item.job_id != job_spec.job_id
        or graph.job_id != job_spec.job_id
        or graph.dataset_job_spec_ref.object_id != job_spec.job_id
        or item.item_id not in graph.item_ids
    ):
        raise ReleaseProjectionPolicyError("release source Job or Item authority is inconsistent")
    if job_spec.approval_policy_ref != approval_ref:
        raise ReleaseProjectionPolicyError("DatasetJobSpec approval policy binding is stale")
    if StageNameV2.RELEASE not in job_spec.requested_stages:
        raise ReleaseProjectionPolicyError("release stage is not enabled")
    try:
        channel = ReleaseChannel(job_spec.export_target.channel)
    except ValueError as exc:
        raise ReleaseProjectionPolicyError("release channel is invalid") from exc
    if (
        channel not in policy.allowed_channels
        or job_spec.export_target.profile not in policy.allowed_export_profiles
        or channel is ReleaseChannel.PRODUCTION
    ):
        raise ReleaseProjectionPolicyError("release target is outside R7-08 policy")
    if item.status not in {ItemStatus.RUNNING, ItemStatus.NEEDS_REVIEW}:
        raise ReleaseProjectionPolicyError("release Item must be running or awaiting review")

    item_index = graph.item_ids.index(item.item_id)
    expected_source_refs = (graph.source_trace_refs[item_index],)
    source_trace_refs = _canonical_refs(source.source_trace_refs)
    if source_trace_refs != expected_source_refs:
        raise ReleaseProjectionPolicyError("release source trace inventory is not exact")
    if len(source_trace_refs) > policy.max_source_trace_refs:
        raise ReleaseProjectionPolicyError("release source trace ref budget exceeded")

    labels = tuple(
        LabelDecisionV2.model_validate(value.model_dump(mode="python")) for value in source.label_decisions
    )
    label_refs = _canonical_refs(tuple(label_decision_ref(value) for value in labels))
    if (
        len(labels) != len(label_refs)
        or len(label_refs) > policy.max_label_decision_refs
        or any(
            value.decision is not LabelDecisionValueV2.MATCH
            or value.execution_status is not LabelExecutionStatus.FINAL
            for value in labels
        )
    ):
        raise ReleaseProjectionPolicyError("release requires unique final matching LabelDecisions")

    _validate_task_chain(draft, gate, contract_set)
    r4_ref = r4_task_contract_set_ref(contract_set)
    item_quality_ref = item_quality_compilation_result_ref(item_quality)
    quality_ref = quality_report_v2_ref(item_quality.quality_report)
    package = item_quality.final_package_manifest
    provenance = item_quality.provenance_manifest
    environment = item_quality.environment_spec
    if (
        not item_quality.quality_report.approvable
        or package is None
        or provenance is None
        or environment is None
        or item_quality.package_sha256 is None
        or item_quality.input_state_only is not True
    ):
        raise ReleaseProjectionPolicyError("release requires an approvable Item quality result")
    if item_quality.quality_report.r4_task_contract_set_ref != r4_ref:
        raise ReleaseProjectionPolicyError("Item quality R4 authority is stale")
    if attachment.producer_task_view_ref != contract_set.producer_task_view_ref:
        raise ReleaseProjectionPolicyError("attachment result differs from current R4 producer view")

    if (
        not batch_quality.approvable
        or batch_quality.open_p0_count
        or batch_quality.open_p1_count
        or batch_quality.unresolved_non_waivable_count
        or batch_quality.invalidated_result_refs
        or item.item_id in batch_quality.blocked_item_ids
    ):
        raise ReleaseProjectionPolicyError("release requires an approvable Batch quality report")
    try:
        batch_index = batch_quality.item_ids.index(item.item_id)
    except ValueError as exc:
        raise ReleaseProjectionPolicyError("Batch quality report does not contain the Item") from exc
    if (
        batch_quality.resolved_job_work_graph_ref != resolved_job_work_graph_v2_ref(graph)
        or batch_quality.item_quality_result_refs[batch_index] != item_quality_ref
        or batch_quality.base_item_quality_report_refs[batch_index] != quality_ref
        or batch_quality.current_item_quality_report_refs[batch_index] != quality_ref
    ):
        raise ReleaseProjectionPolicyError("Batch quality Item bindings are stale")
    lineage = batch_quality.lineage_audits[batch_index]
    if (
        lineage.item_id != item.item_id
        or lineage.source_trace_ref != source_trace_refs[0]
        or lineage.label_decision_refs != label_refs
        or lineage.task_draft_ref != task_draft_ref(draft)
        or lineage.r4_task_contract_set_ref != r4_ref
        or lineage.item_quality_result_ref != item_quality_ref
        or lineage.base_quality_report_ref != quality_ref
        or lineage.final_package_manifest_ref != final_package_manifest_ref(package)
        or lineage.provenance_manifest_ref != provenance_manifest_v2_ref(provenance)
        or lineage.environment_spec_ref != environment_spec_v2_ref(environment)
    ):
        raise ReleaseProjectionPolicyError("Batch quality lineage differs from release source")

    query = _query_spec(draft, audit=source.job_spec.audit)
    components = EvaluationItemComponents(
        query_spec_ref=query_spec_ref(query),
        environment_spec_ref=environment_spec_v2_ref(environment),
        rubric_set_ref=contract_set.rubric_set_ref,
        evaluator_spec_ref=contract_set.evaluator_spec_ref,
        reference_policy_ref=contract_set.reference_policy_ref,
        tool_policy_ref=contract_set.tool_policy_ref,
        provenance_manifest_ref=provenance_manifest_v2_ref(provenance),
        quality_report_ref=quality_ref,
    )

    not_required = _canonical_checkpoints(source.not_required_checkpoints)
    if any(value is not ApprovalCheckpoint.ENVIRONMENT_STRATEGY for value in not_required) or not set(
        not_required
    ).issubset(approval.enabled_checkpoints):
        raise ReleaseProjectionPolicyError("only enabled Environment Strategy may be not required")
    required = _canonical_checkpoints(
        tuple(value for value in approval.enabled_checkpoints if value not in not_required)
    )
    decision_refs, plan_bindings = _candidate_decisions(
        decisions,
        source=source,
        required_checkpoints=required,
        approval_policy_ref=approval_ref,
        policy=policy,
        allow_final_decision=allow_final_decision,
    )
    applications, reports, report_refs, current_heads = _revalidation_authority(
        source,
        item_id=item.item_id,
        policy=policy,
    )
    return _PreparedReleaseSource(
        source=EvaluationItemReleaseSource(
            job_spec=job_spec,
            item=item,
            resolved_job_work_graph=graph,
            source_trace_refs=source_trace_refs,
            label_decisions=labels,
            task_draft=draft,
            task_prompt_safety_gate=gate,
            task_contract_set=contract_set,
            attachment_result=attachment,
            item_quality=item_quality,
            batch_quality=batch_quality,
            approval_policy=approval,
            decision_commits=decisions,
            relevant_revalidation_application_refs=applications,
            revalidation_reports=reports,
            current_head_refs=current_heads,
            not_required_checkpoints=not_required,
        ),
        query_spec=query,
        components=components,
        required_checkpoints=required,
        not_required_checkpoints=not_required,
        plan_bindings=plan_bindings,
        current_decision_refs=decision_refs,
        relevant_application_refs=applications,
        report_refs=report_refs,
        current_head_refs=current_heads,
    )


def _validate_task_chain(
    draft: TaskDraftV2,
    gate: TaskPromptSafetyGateV2,
    contract_set: R4TaskContractSetV2,
) -> None:
    draft_digest = task_draft_carried_sha256(draft)
    if (
        draft.task_draft_sha256 != draft_digest
        or draft.task_draft_id != f"task-draft://sha256/{draft_digest}"
    ):
        raise ReleaseProjectionPolicyError("TaskDraft identity is stale")
    contract_digest = r4_task_contract_set_carried_sha256(contract_set)
    if (
        contract_set.contract_set_sha256 != contract_digest
        or contract_set.contract_set_id != f"r4-task-contract-set://sha256/{contract_digest}"
    ):
        raise ReleaseProjectionPolicyError("R4 contract set identity is stale")
    if (
        draft.prompt_safety_status is not TaskDraftPromptSafetyStatusV2.PASSED
        or gate.status is not TaskPromptSafetyGateStatusV2.PASSED
        or draft.prompt_safety_gate_ref != task_prompt_safety_gate_ref(gate)
        or contract_set.task_draft_ref != task_draft_ref(draft)
        or contract_set.task_prompt_safety_gate_ref != task_prompt_safety_gate_ref(gate)
    ):
        raise ReleaseProjectionPolicyError("R4 TaskDraft and prompt safety authority is stale")


def _candidate_decisions(
    decisions: tuple[UserDecisionCommitResultV2, ...],
    *,
    source: EvaluationItemReleaseSource,
    required_checkpoints: tuple[ApprovalCheckpoint, ...],
    approval_policy_ref: ObjectRef,
    policy: ReleaseProjectionPolicyV2,
    allow_final_decision: bool,
) -> tuple[tuple[ObjectRef, ...], tuple[CheckpointDecisionBinding, ...]]:
    refs = tuple(user_decision_commit_result_v2_ref(value) for value in decisions)
    if refs != _canonical_refs(refs):
        raise ReleaseProjectionPolicyError("current user decision commits must be canonical")
    if len(decisions) > policy.max_user_decision_refs:
        raise ReleaseProjectionPolicyError("release user decision ref budget exceeded")
    by_checkpoint: dict[ApprovalCheckpoint, UserDecisionCommitResultV2] = {}
    decision_refs: list[ObjectRef] = []
    for commit in decisions:
        record = commit.decision_record
        if (
            commit.job_id != source.item.job_id
            or record.approval_policy_ref != approval_policy_ref
            or record.checkpoint not in source.approval_policy.enabled_checkpoints
        ):
            raise ReleaseProjectionPolicyError("current user decision authority is cross-Job or stale")
        if record.checkpoint in by_checkpoint:
            raise ReleaseProjectionPolicyError("current user decisions contain duplicate checkpoints")
        by_checkpoint[record.checkpoint] = commit
        decision_refs.append(commit.decision_record_ref)
    if ApprovalCheckpoint.FINAL_DATASET_REVIEW in by_checkpoint and not allow_final_decision:
        raise ReleaseProjectionPolicyError("candidate cannot consume a Final decision for another subject")
    bindings: list[CheckpointDecisionBinding] = []
    for checkpoint in required_checkpoints:
        if checkpoint is ApprovalCheckpoint.FINAL_DATASET_REVIEW:
            continue
        selected = by_checkpoint.get(checkpoint)
        if selected is None or selected.outcome is not UserDecisionCommitOutcomeV2.ACCEPTED:
            raise ReleaseProjectionPolicyError("every required plan checkpoint must be currently accepted")
        bindings.append(_checkpoint_binding(selected))
    return _canonical_refs(tuple(decision_refs)), _canonical_bindings(tuple(bindings))


def _revalidation_authority(
    source: EvaluationItemReleaseSource,
    *,
    item_id: str,
    policy: ReleaseProjectionPolicyV2,
) -> tuple[
    tuple[ObjectRef, ...],
    tuple[DirectedRevalidationReportV2, ...],
    tuple[ObjectRef, ...],
    tuple[ObjectRef, ...],
]:
    applications = _canonical_refs(source.relevant_revalidation_application_refs)
    if applications != source.relevant_revalidation_application_refs:
        raise ReleaseProjectionPolicyError("relevant revalidation applications must be canonical")
    try:
        reports = tuple(
            DirectedRevalidationReportV2.model_validate(value.model_dump(mode="python"))
            for value in source.revalidation_reports
        )
        for report in reports:
            validate_directed_revalidation_report_v2_identity(report)
    except (ValidationError, ValueError) as exc:
        raise ReleaseProjectionPolicyError("revalidation report authority is stale") from exc
    report_refs = tuple(directed_revalidation_report_v2_ref(value) for value in reports)
    if report_refs != _canonical_refs(report_refs):
        raise ReleaseProjectionPolicyError("revalidation reports must be canonical")
    if (
        len(reports) > policy.max_revalidation_reports
        or len(applications) != len(reports)
        or _canonical_refs(tuple(value.application_ref for value in reports)) != applications
    ):
        raise ReleaseProjectionPolicyError("every relevant application requires one current report")
    current_heads = _canonical_refs(source.current_head_refs)
    if current_heads != source.current_head_refs:
        raise ReleaseProjectionPolicyError("release current heads must be canonical")
    if len(current_heads) > policy.max_current_head_refs:
        raise ReleaseProjectionPolicyError("release current head ref budget exceeded")
    current_set = set(current_heads)
    for report in reports:
        if report.outcome is not DirectedRevalidationReportOutcomeV2.COMPLETE:
            raise ReleaseProjectionPolicyError("every relevant application requires a COMPLETE report")
        if item_id in report.excluded_item_ids:
            raise ReleaseProjectionPolicyError("revalidation excluded the release Item")
        if not set(report.replacement_current_refs).issubset(current_set):
            raise ReleaseProjectionPolicyError("revalidation replacement heads are not current")
        if set(report.invalidated_prior_refs).intersection(current_set):
            raise ReleaseProjectionPolicyError("invalidated revalidation refs remain current")
    return applications, reports, report_refs, current_heads


def _query_spec(
    draft: TaskDraftV2,
    *,
    audit: ContractAudit,
) -> QuerySpec:
    dependencies = tuple(sorted(item.dependency_id for item in draft.attachment_dependencies))
    prompt_sha256 = hashlib.sha256(draft.visible_prompt.encode()).hexdigest()
    value = QuerySpec(
        query_spec_id="query-spec://pending",
        task_draft_ref=task_draft_ref(draft),
        prompt=draft.visible_prompt,
        attachment_dependency_ids=dependencies,
        prompt_sha256=prompt_sha256,
        audit=audit.model_copy(update={"input_refs": (task_draft_ref(draft),)}),
    )
    digest = query_spec_carried_sha256(value)
    return value.model_copy(update={"query_spec_id": f"query-spec://sha256/{digest}"})


def _release_subject(
    prepared: _PreparedReleaseSource,
    *,
    checkpoint_bindings: tuple[CheckpointDecisionBinding, ...],
    audit: ContractAudit,
    policy: ReleaseProjectionPolicyV2,
) -> EvaluationItemReleaseSubjectV2:
    source = prepared.source
    item_quality = source.item_quality
    assert item_quality.final_package_manifest is not None
    return EvaluationItemReleaseSubjectV2.create(
        job_id=source.item.job_id,
        item_id=source.item.item_id,
        source_trace_refs=source.source_trace_refs,
        label_decision_refs=tuple(label_decision_ref(value) for value in source.label_decisions),
        task_draft_ref=task_draft_ref(source.task_draft),
        attachment_reconstruction_result_ref=(
            attachment_reconstruction_result_v2_ref(source.attachment_result)
        ),
        components=prepared.components,
        item_quality_result_ref=item_quality_compilation_result_ref(item_quality),
        batch_quality_report_ref=batch_quality_report_v2_ref(source.batch_quality),
        package_manifest_ref=final_package_manifest_ref(item_quality.final_package_manifest),
        package_sha256=item_quality.package_sha256 or "",
        user_approval_policy_ref=user_approval_policy_ref(source.approval_policy),
        required_checkpoints=prepared.required_checkpoints,
        not_required_checkpoints=prepared.not_required_checkpoints,
        satisfied_checkpoint_bindings=checkpoint_bindings,
        user_decision_record_refs=prepared.current_decision_refs,
        relevant_revalidation_application_refs=(prepared.relevant_application_refs),
        revalidation_report_refs=prepared.report_refs,
        current_head_refs=prepared.current_head_refs,
        channel=ReleaseChannel(source.job_spec.export_target.channel),
        registry=source.job_spec.export_target.registry,
        export_profile="LH",
        export_profile_version=source.job_spec.export_target.profile_version,
        policy_ref=policy.to_ref(),
        audit=audit,
    )


def _terminal_action(
    prepared: _PreparedReleaseSource,
    *,
    candidate_item_ref: ObjectRef,
) -> tuple[ReleaseActionV2, tuple[CheckpointDecisionBinding, ...]]:
    required = prepared.required_checkpoints
    by_checkpoint = {value.decision_record.checkpoint: value for value in prepared.source.decision_commits}
    bindings = list(prepared.plan_bindings)
    if ApprovalCheckpoint.FINAL_DATASET_REVIEW in required:
        final = by_checkpoint.get(ApprovalCheckpoint.FINAL_DATASET_REVIEW)
        if final is None:
            raise ReleaseProjectionPendingError("Final Dataset Review is still pending")
        if final.decision_record.subject_refs != (candidate_item_ref,):
            raise ReleaseProjectionPolicyError("Final Dataset Review targets a stale candidate Item")
        if final.outcome is UserDecisionCommitOutcomeV2.ACCEPTED:
            bindings.append(_checkpoint_binding(final))
            return ReleaseActionV2.APPROVE, _canonical_bindings(tuple(bindings))
        if final.outcome is UserDecisionCommitOutcomeV2.REJECTED:
            return ReleaseActionV2.REJECT, _canonical_bindings(tuple(bindings))
        raise ReleaseProjectionPendingError(
            "Final Dataset Review remains pending after a non-terminal decision"
        )
    if any(
        value.outcome is UserDecisionCommitOutcomeV2.REJECTED for value in prepared.source.decision_commits
    ):
        return ReleaseActionV2.REJECT, _canonical_bindings(tuple(bindings))
    if {value.checkpoint for value in bindings} != set(required):
        raise ReleaseProjectionPendingError("required release checkpoints remain pending")
    return ReleaseActionV2.APPROVE, _canonical_bindings(tuple(bindings))


def _checkpoint_binding(
    value: UserDecisionCommitResultV2,
) -> CheckpointDecisionBinding:
    return CheckpointDecisionBinding(
        checkpoint=value.decision_record.checkpoint,
        request_ref=value.request_ref,
        decision_record_ref=value.decision_record_ref,
    )


def _release_decision(
    *,
    subject: EvaluationItemReleaseSubjectV2,
    action: ReleaseActionV2,
    chain_id: str,
    revision: int,
    previous_decision_ref: ObjectRef | None,
    checkpoint_bindings: tuple[CheckpointDecisionBinding, ...],
    audit: ContractAudit,
) -> ReleaseDecisionV2:
    state = {
        ReleaseActionV2.REQUEST_RELEASE: ReleaseStateV2.CANDIDATE,
        ReleaseActionV2.APPROVE: ReleaseStateV2.APPROVED,
        ReleaseActionV2.REJECT: ReleaseStateV2.REJECTED,
    }[action]
    refs = (
        subject.to_ref(),
        subject.package_manifest_ref,
        subject.batch_quality_report_ref,
        *(
            ref
            for binding in checkpoint_bindings
            for ref in (binding.request_ref, binding.decision_record_ref)
        ),
        *((previous_decision_ref,) if previous_decision_ref else ()),
    )
    value = ReleaseDecisionV2(
        release_decision_id="release-decision://pending",
        chain_id=chain_id,
        previous_decision_ref=previous_decision_ref,
        item_id=subject.item_id,
        item_version=str(revision),
        components=subject.components,
        release_subject_sha256=subject.release_subject_sha256,
        package_manifest_ref=subject.package_manifest_ref,
        package_sha256=subject.package_sha256,
        batch_quality_report_ref=subject.batch_quality_report_ref,
        automated_quality_passed=True,
        open_p0_count=0,
        open_p1_count=0,
        unresolved_non_waivable_count=0,
        user_approval_policy_ref=subject.user_approval_policy_ref,
        required_checkpoints=frozenset(subject.required_checkpoints),
        checkpoint_decisions=checkpoint_bindings,
        channel=subject.channel,
        registry=subject.registry,
        export_profile=subject.export_profile,
        export_profile_version=subject.export_profile_version,
        production_attestation_ref=None,
        action=action,
        state=state,
        idempotency_key=(
            f"release-{action.value.casefold().replace('_', '-')}-{subject.release_subject_sha256[:24]}"
        ),
        actor=audit.created_by,
        decided_at=audit.created_at,
        decision_sha256="0" * 64,
        audit=audit.model_copy(update={"input_refs": _canonical_refs(refs)}),
    )
    digest = release_decision_v2_carried_sha256(value)
    return value.model_copy(
        update={
            "release_decision_id": f"release-decision://sha256/{digest}",
            "decision_sha256": digest,
        }
    )


def _evaluation_item(
    *,
    subject: EvaluationItemReleaseSubjectV2,
    decision: ReleaseDecisionV2,
    revision: int,
    audit: ContractAudit,
    user_decision_record_refs: tuple[ObjectRef, ...] | None = None,
) -> EvaluationItemV2:
    decision_ref = _release_decision_ref(decision)
    decision_record_refs = (
        subject.user_decision_record_refs
        if user_decision_record_refs is None
        else _canonical_refs(user_decision_record_refs)
    )
    refs = (
        subject.to_ref(),
        *subject.source_trace_refs,
        *subject.label_decision_refs,
        subject.task_draft_ref,
        subject.attachment_reconstruction_result_ref,
        *decision_record_refs,
        decision_ref,
    )
    value = EvaluationItemV2(
        evaluation_item_id="evaluation-item://pending",
        item_version=str(revision),
        source_trace_refs=subject.source_trace_refs,
        label_decision_refs=subject.label_decision_refs,
        task_draft_ref=subject.task_draft_ref,
        attachment_reconstruction_result_ref=(subject.attachment_reconstruction_result_ref),
        components=subject.components,
        user_approval_policy_ref=subject.user_approval_policy_ref,
        user_decision_record_refs=decision_record_refs,
        release_decision_ref=decision_ref,
        item_sha256="0" * 64,
        audit=audit.model_copy(update={"input_refs": _canonical_refs(refs)}),
    )
    digest = evaluation_item_v2_carried_sha256(value)
    return value.model_copy(
        update={
            "evaluation_item_id": f"evaluation-item://sha256/{digest}",
            "item_sha256": digest,
        }
    )


def _require_same_candidate_authority(
    candidate: ReleaseProjectionResultV2,
    prepared: _PreparedReleaseSource,
) -> None:
    source = prepared.source
    subject = candidate.release_subject
    item_quality = source.item_quality
    assert item_quality.final_package_manifest is not None
    expected = (
        subject.job_id == source.item.job_id,
        subject.item_id == source.item.item_id,
        subject.source_trace_refs == source.source_trace_refs,
        subject.label_decision_refs == tuple(label_decision_ref(value) for value in source.label_decisions),
        subject.task_draft_ref == task_draft_ref(source.task_draft),
        subject.attachment_reconstruction_result_ref
        == attachment_reconstruction_result_v2_ref(source.attachment_result),
        subject.components == prepared.components,
        subject.item_quality_result_ref == item_quality_compilation_result_ref(item_quality),
        subject.batch_quality_report_ref == batch_quality_report_v2_ref(source.batch_quality),
        subject.package_manifest_ref == final_package_manifest_ref(item_quality.final_package_manifest),
        subject.package_sha256 == item_quality.package_sha256,
        subject.user_approval_policy_ref == user_approval_policy_ref(source.approval_policy),
        subject.required_checkpoints == prepared.required_checkpoints,
        subject.not_required_checkpoints == prepared.not_required_checkpoints,
        subject.satisfied_checkpoint_bindings == prepared.plan_bindings,
        subject.user_decision_record_refs
        == _canonical_refs(tuple(binding.decision_record_ref for binding in prepared.plan_bindings)),
        subject.relevant_revalidation_application_refs == prepared.relevant_application_refs,
        subject.revalidation_report_refs == prepared.report_refs,
        subject.current_head_refs == prepared.current_head_refs,
    )
    if not all(expected) or candidate.query_spec_ref != query_spec_ref(prepared.query_spec):
        raise ReleaseProjectionPolicyError("candidate differs from current release authority")


def _canonical_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    by_key = {
        (
            value.object_type,
            value.object_id,
            value.object_version,
            value.object_sha256,
        ): value
        for value in values
    }
    return tuple(by_key[key] for key in sorted(by_key))


def _canonical_checkpoints(
    values: tuple[ApprovalCheckpoint, ...],
) -> tuple[ApprovalCheckpoint, ...]:
    order = {value: index for index, value in enumerate(ApprovalCheckpoint)}
    return tuple(sorted(set(values), key=lambda value: order[value]))


def _canonical_bindings(
    values: tuple[CheckpointDecisionBinding, ...],
) -> tuple[CheckpointDecisionBinding, ...]:
    order = {value: index for index, value in enumerate(ApprovalCheckpoint)}
    return tuple(sorted(values, key=lambda value: order[value.checkpoint]))


def _chain_id(job_id: str, item_id: str) -> str:
    digest = hashlib.sha256(f"{job_id}:{item_id}".encode()).hexdigest()
    return f"release-chain://sha256/{digest}"


def _release_decision_ref(value: ReleaseDecisionV2) -> ObjectRef:
    from eval_factory.contracts.release_projection_v2 import (
        release_decision_v2_ref,
    )

    return release_decision_v2_ref(value)


def _evaluation_item_ref(value: EvaluationItemV2) -> ObjectRef:
    from eval_factory.contracts.release_projection_v2 import evaluation_item_v2_ref

    return evaluation_item_v2_ref(value)
