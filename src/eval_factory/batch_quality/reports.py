from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass

from pydantic import ValidationError

from eval_factory.batch_quality.cross_item_safety import (
    CrossItemSafetyCompiler,
    CrossItemSafetyItemSource,
    CrossItemSafetyPolicyError,
)
from eval_factory.batch_quality.duplicates import (
    DuplicateDetectionCompiler,
    DuplicateDetectionItemSource,
    DuplicateDetectionPolicyError,
)
from eval_factory.contracts.batch_quality_v2 import (
    BATCH_QUALITY_ITEM_LINEAGE_CHECKS,
    BatchFindingCategoryV2,
    BatchFindingScopeV2,
    BatchQualityPolicyV2,
    BatchQualityReportV2,
    BatchReviewerFindingV2,
    ItemLineageAuditResultV2,
    ItemQualityReportRevisionV2,
    LineageAuditCheckCodeV2,
    LineageAuditCheckOutcomeV2,
    LineageAuditCheckV2,
    batch_quality_policy_v2_ref,
    batch_quality_report_v2_ref,
    batch_reviewer_finding_v2_ref,
    item_quality_report_revision_v2_ref,
    lineage_audit_check_v2_ref,
    validate_batch_quality_policy_v2_identity,
    validate_batch_quality_report_v2_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
)
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.cross_item_safety_v2 import (
    AnswerReusePairEvidenceV2,
    CrossItemSafetyEvidenceKindV2,
    CrossItemSafetyPolicyV2,
    CrossItemSafetyResultV2,
    answer_reuse_pair_evidence_v2_ref,
    cross_item_safety_cluster_v2_ref,
    cross_item_safety_policy_v2_ref,
    cross_item_visible_match_evidence_v2_ref,
)
from eval_factory.contracts.duplicate_v2 import (
    DuplicateDetectionPolicyV2,
    DuplicateDetectionResultV2,
    DuplicateMatchKindV2,
    DuplicateSubjectKindV2,
    duplicate_cluster_v2_ref,
    duplicate_detection_policy_v2_ref,
    duplicate_pair_evidence_v2_ref,
)
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
    LabelDecisionValueV2,
    LabelExecutionStatus,
    SelectionContextV2,
    label_decision_ref,
    selection_context_ref,
)
from eval_factory.contracts.orchestration_v2 import (
    ResolvedJobWorkGraphV2,
    resolved_job_work_graph_v2_ref,
    validate_resolved_job_work_graph_v2_identity,
)
from eval_factory.contracts.quality_v2 import (
    ItemQualityCompilationResultV2,
    environment_spec_v2_ref,
    final_package_manifest_ref,
    item_quality_compilation_result_ref,
    package_member_provenance_ref,
    provenance_manifest_v2_ref,
    quality_report_v2_ref,
)
from eval_factory.contracts.task_v2 import (
    PromptLeakageReferenceSetV2,
    R4TaskContractSetV2,
    TaskDraftV2,
    TaskPromptSafetyGateV2,
    prompt_leakage_reference_set_ref,
    r4_task_contract_set_ref,
    task_draft_carried_sha256,
    task_draft_ref,
)
from eval_factory.contracts.trace import TraceEnvelope
from eval_factory.labeling import LABEL_DECISION_MERGE_POLICY_VERSION
from eval_factory.task_authoring import (
    SELECTION_CONTEXT_FIREWALL_POLICY_VERSION,
)


class BatchQualityPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BatchQualityItemSource:
    item_id: str
    source_trace_ref: ObjectRef
    trace_envelope: TraceEnvelope
    label_decisions: tuple[LabelDecisionV2, ...]
    selection_context: SelectionContextV2
    task_draft: TaskDraftV2
    task_prompt_safety_gate: TaskPromptSafetyGateV2
    leakage_reference_set: PromptLeakageReferenceSetV2
    task_contract_set: R4TaskContractSetV2
    item_quality: ItemQualityCompilationResultV2


@dataclass(frozen=True, slots=True)
class _AdmittedSources:
    graph: ResolvedJobWorkGraphV2
    graph_ref: ObjectRef
    policy: BatchQualityPolicyV2
    policy_ref: ObjectRef
    sources: tuple[BatchQualityItemSource, ...]


class BatchQualityCompiler:
    def compile(
        self,
        *,
        resolved_job_work_graph: ResolvedJobWorkGraphV2,
        policy: BatchQualityPolicyV2,
        sources: tuple[BatchQualityItemSource, ...],
        duplicate_policy: DuplicateDetectionPolicyV2,
        duplicate_result: DuplicateDetectionResultV2,
        cross_item_policy: CrossItemSafetyPolicyV2,
        cross_item_result: CrossItemSafetyResultV2,
        audit: ContractAudit,
    ) -> BatchQualityReportV2:
        admitted = self._admit(
            resolved_job_work_graph=resolved_job_work_graph,
            policy=policy,
            sources=sources,
        )
        self._validate_r7_results(
            admitted=admitted,
            duplicate_policy=duplicate_policy,
            duplicate_result=duplicate_result,
            cross_item_policy=cross_item_policy,
            cross_item_result=cross_item_result,
        )
        lineage_audits = tuple(
            _compile_item_lineage(
                source,
                policy_ref=admitted.policy_ref,
                audit=audit,
            )
            for source in admitted.sources
        )
        batch_lineage_check = _compile_batch_alias_check(
            admitted,
            audit=audit,
        )
        findings = _compile_findings(
            duplicate_result=duplicate_result,
            cross_item_result=cross_item_result,
            lineage_audits=lineage_audits,
            batch_lineage_check=batch_lineage_check,
            policy_ref=admitted.policy_ref,
            audit=audit,
        )
        _validate_budgets(
            admitted,
            lineage_audits=lineage_audits,
            batch_lineage_check=batch_lineage_check,
            findings=findings,
        )
        revisions = _compile_revisions(
            admitted,
            findings=findings,
            audit=audit,
        )
        base_quality_refs = tuple(source.item_quality.quality_report_ref for source in admitted.sources)
        revision_by_item = {value.item_id: value for value in revisions}
        current_quality_refs = tuple(
            (
                item_quality_report_revision_v2_ref(revision_by_item[source.item_id])
                if source.item_id in revision_by_item
                else source.item_quality.quality_report_ref
            )
            for source in admitted.sources
        )
        duplicate_pair_refs = tuple(
            duplicate_pair_evidence_v2_ref(value) for value in duplicate_result.duplicate_pairs
        )
        duplicate_cluster_refs = tuple(
            duplicate_cluster_v2_ref(value) for value in duplicate_result.duplicate_clusters
        )
        visible_match_refs = tuple(
            cross_item_visible_match_evidence_v2_ref(value) for value in cross_item_result.visible_matches
        )
        answer_reuse_refs = tuple(
            answer_reuse_pair_evidence_v2_ref(value) for value in cross_item_result.answer_reuse_pairs
        )
        safety_cluster_refs = tuple(
            cross_item_safety_cluster_v2_ref(value) for value in cross_item_result.clusters
        )
        return BatchQualityReportV2.create(
            batch_id=admitted.graph.job_id,
            resolved_job_work_graph_ref=admitted.graph_ref,
            policy_ref=admitted.policy_ref,
            duplicate_policy_ref=duplicate_detection_policy_v2_ref(duplicate_policy),
            duplicate_result_ref=ObjectRef(
                object_type="duplicate-detection-result",
                object_id=duplicate_result.duplicate_detection_result_id,
                object_version="v2",
                object_sha256=duplicate_result.result_sha256,
            ),
            cross_item_policy_ref=cross_item_safety_policy_v2_ref(cross_item_policy),
            cross_item_result_ref=ObjectRef(
                object_type="cross-item-safety-result",
                object_id=cross_item_result.cross_item_safety_result_id,
                object_version="v2",
                object_sha256=cross_item_result.result_sha256,
            ),
            item_ids=tuple(source.item_id for source in admitted.sources),
            item_quality_result_refs=tuple(
                item_quality_compilation_result_ref(source.item_quality) for source in admitted.sources
            ),
            base_item_quality_report_refs=base_quality_refs,
            current_item_quality_report_refs=current_quality_refs,
            lineage_audits=lineage_audits,
            batch_lineage_checks=(batch_lineage_check,),
            findings=findings,
            item_quality_report_revisions=revisions,
            duplicate_pair_refs=duplicate_pair_refs,
            duplicate_cluster_refs=duplicate_cluster_refs,
            visible_match_refs=visible_match_refs,
            answer_reuse_pair_refs=answer_reuse_refs,
            safety_cluster_refs=safety_cluster_refs,
            invalidated_result_refs=(),
            audit=audit,
        )

    def validate_current(
        self,
        report: BatchQualityReportV2,
        *,
        resolved_job_work_graph: ResolvedJobWorkGraphV2,
        policy: BatchQualityPolicyV2,
        sources: tuple[BatchQualityItemSource, ...],
        duplicate_policy: DuplicateDetectionPolicyV2,
        duplicate_result: DuplicateDetectionResultV2,
        cross_item_policy: CrossItemSafetyPolicyV2,
        cross_item_result: CrossItemSafetyResultV2,
    ) -> None:
        try:
            validate_batch_quality_report_v2_identity(report)
        except ValueError as exc:
            raise BatchQualityPolicyError("BatchQualityReport identity is stale") from exc
        rebuilt = self.compile(
            resolved_job_work_graph=resolved_job_work_graph,
            policy=policy,
            sources=sources,
            duplicate_policy=duplicate_policy,
            duplicate_result=duplicate_result,
            cross_item_policy=cross_item_policy,
            cross_item_result=cross_item_result,
            audit=report.audit,
        )
        if batch_quality_report_v2_ref(rebuilt) != (batch_quality_report_v2_ref(report)):
            raise BatchQualityPolicyError("BatchQualityReport is not current")

    def _admit(
        self,
        *,
        resolved_job_work_graph: ResolvedJobWorkGraphV2,
        policy: BatchQualityPolicyV2,
        sources: tuple[BatchQualityItemSource, ...],
    ) -> _AdmittedSources:
        try:
            graph = ResolvedJobWorkGraphV2.model_validate(resolved_job_work_graph.model_dump(mode="python"))
            validate_resolved_job_work_graph_v2_identity(graph)
        except (ValidationError, ValueError) as exc:
            raise BatchQualityPolicyError("resolved Job work graph is stale or malformed") from exc
        try:
            parsed_policy = BatchQualityPolicyV2.model_validate(policy.model_dump(mode="python"))
            validate_batch_quality_policy_v2_identity(parsed_policy)
        except (ValidationError, ValueError) as exc:
            raise BatchQualityPolicyError("batch quality policy is stale or malformed") from exc
        if len(sources) < 2:
            raise BatchQualityPolicyError("batch quality requires at least two Items")
        if len(sources) > parsed_policy.max_items:
            raise BatchQualityPolicyError("batch quality Item limit exceeded")
        expected_by_item = dict(
            zip(
                graph.item_ids,
                graph.source_trace_refs,
                strict=True,
            )
        )
        item_ids = tuple(source.item_id for source in sources)
        if len(item_ids) != len(set(item_ids)):
            raise BatchQualityPolicyError("duplicate batch quality Item")
        if set(item_ids) != set(graph.item_ids):
            raise BatchQualityPolicyError("batch quality Item inventory does not match graph")
        admitted: list[BatchQualityItemSource] = []
        for source in sources:
            expected_trace = expected_by_item[source.item_id]
            if source.source_trace_ref != expected_trace:
                raise BatchQualityPolicyError("batch quality source trace does not match graph")
            admitted.append(_parse_source(source))
        ordered = tuple(sorted(admitted, key=lambda value: value.item_id))
        if sum(len(value.label_decisions) for value in ordered) > (parsed_policy.max_label_decisions):
            raise BatchQualityPolicyError("batch quality label decision limit exceeded")
        package_member_count = sum(
            len(value.item_quality.final_package_manifest.entries)
            for value in ordered
            if value.item_quality.final_package_manifest is not None
        )
        if package_member_count > parsed_policy.max_package_members:
            raise BatchQualityPolicyError("batch quality package member limit exceeded")
        _require_unique_refs(
            "TaskDraft ownership",
            tuple(task_draft_ref(value.task_draft) for value in ordered),
        )
        _require_unique_refs(
            "R4 contract-set ownership",
            tuple(r4_task_contract_set_ref(value.task_contract_set) for value in ordered),
        )
        _require_unique_refs(
            "Item quality ownership",
            tuple(item_quality_compilation_result_ref(value.item_quality) for value in ordered),
        )
        return _AdmittedSources(
            graph=graph,
            graph_ref=resolved_job_work_graph_v2_ref(graph),
            policy=parsed_policy,
            policy_ref=batch_quality_policy_v2_ref(parsed_policy),
            sources=ordered,
        )

    @staticmethod
    def _validate_r7_results(
        *,
        admitted: _AdmittedSources,
        duplicate_policy: DuplicateDetectionPolicyV2,
        duplicate_result: DuplicateDetectionResultV2,
        cross_item_policy: CrossItemSafetyPolicyV2,
        cross_item_result: CrossItemSafetyResultV2,
    ) -> None:
        duplicate_sources = tuple(
            DuplicateDetectionItemSource(
                item_id=source.item_id,
                task_draft=source.task_draft,
                task_contract_set=source.task_contract_set,
                item_quality=source.item_quality,
            )
            for source in admitted.sources
        )
        cross_sources = tuple(
            CrossItemSafetyItemSource(
                item_id=source.item_id,
                task_draft=source.task_draft,
                task_prompt_safety_gate=source.task_prompt_safety_gate,
                leakage_reference_set=source.leakage_reference_set,
                task_contract_set=source.task_contract_set,
                item_quality=source.item_quality,
            )
            for source in admitted.sources
        )
        try:
            DuplicateDetectionCompiler().validate_current(
                duplicate_result,
                resolved_job_work_graph=admitted.graph,
                policy=duplicate_policy,
                sources=duplicate_sources,
            )
        except DuplicateDetectionPolicyError as exc:
            raise BatchQualityPolicyError("duplicate detection result is not current") from exc
        try:
            CrossItemSafetyCompiler().validate_current(
                cross_item_result,
                resolved_job_work_graph=admitted.graph,
                policy=cross_item_policy,
                sources=cross_sources,
            )
        except CrossItemSafetyPolicyError as exc:
            raise BatchQualityPolicyError("cross-item safety result is not current") from exc


def _parse_source(source: BatchQualityItemSource) -> BatchQualityItemSource:
    try:
        envelope = TraceEnvelope.model_validate(source.trace_envelope.model_dump(mode="python"))
        labels = tuple(
            LabelDecisionV2.model_validate(value.model_dump(mode="python"))
            for value in source.label_decisions
        )
        selection = SelectionContextV2.model_validate(source.selection_context.model_dump(mode="python"))
        draft = TaskDraftV2.model_validate(source.task_draft.model_dump(mode="python"))
        gate = TaskPromptSafetyGateV2.model_validate(source.task_prompt_safety_gate.model_dump(mode="python"))
        reference_set = PromptLeakageReferenceSetV2.model_validate(
            source.leakage_reference_set.model_dump(mode="python")
        )
        contract_set = R4TaskContractSetV2.model_validate(source.task_contract_set.model_dump(mode="python"))
        item_quality = ItemQualityCompilationResultV2.model_validate(
            source.item_quality.model_dump(mode="python")
        )
    except ValidationError as exc:
        raise BatchQualityPolicyError("batch quality source is malformed") from exc
    if not labels:
        raise BatchQualityPolicyError("batch quality source requires LabelDecision values")
    label_ids = tuple(value.label_decision_id for value in labels)
    if len(label_ids) != len(set(label_ids)):
        raise BatchQualityPolicyError("duplicate LabelDecision identity")
    for label in labels:
        if (
            label.policy_version != LABEL_DECISION_MERGE_POLICY_VERSION
            or label.label_decision_id != f"label-decision://sha256/{label.decision_sha256}"
            or label.execution_status is not LabelExecutionStatus.FINAL
            or label.decision is not LabelDecisionValueV2.MATCH
            or label.unresolved_reasons
        ):
            raise BatchQualityPolicyError("LabelDecision is not current and selectable")
    _validate_selection_context_identity(selection)
    draft_digest = task_draft_carried_sha256(draft)
    if (
        draft.task_draft_sha256 != draft_digest
        or draft.task_draft_id != f"task-draft://sha256/{draft_digest}"
    ):
        raise BatchQualityPolicyError("TaskDraft identity is stale")
    return BatchQualityItemSource(
        item_id=source.item_id,
        source_trace_ref=source.source_trace_ref,
        trace_envelope=envelope,
        label_decisions=tuple(sorted(labels, key=lambda value: _ref_key(label_decision_ref(value)))),
        selection_context=selection,
        task_draft=draft,
        task_prompt_safety_gate=gate,
        leakage_reference_set=reference_set,
        task_contract_set=contract_set,
        item_quality=item_quality,
    )


def _validate_selection_context_identity(
    context: SelectionContextV2,
) -> None:
    if context.policy_version != SELECTION_CONTEXT_FIREWALL_POLICY_VERSION:
        raise BatchQualityPolicyError("selection context policy is stale")
    payload = {
        "candidate_id": context.candidate_id,
        "approved_label_decision_refs": [_ref_payload(ref) for ref in context.approved_label_decision_refs],
        "safe_evidence_bundle_ref": _ref_payload(context.safe_evidence_bundle_ref),
        "task_authoring_note_refs": [_ref_payload(ref) for ref in context.task_authoring_note_refs],
        "excluded_signal_hashes": list(context.excluded_signal_hashes),
        "projection_policy_ref": _ref_payload(context.projection_policy_ref),
        "policy_version": context.policy_version,
    }
    digest = _payload_sha256(payload)
    if (
        context.selection_context_sha256 != digest
        or context.selection_context_id != f"selection-context://sha256/{digest}"
    ):
        raise BatchQualityPolicyError("selection context identity is stale")


def _trace_envelope_ref(envelope: TraceEnvelope) -> ObjectRef:
    return ObjectRef(
        object_type="trace-envelope",
        object_id=envelope.trace_ir_version_id,
        object_version="stored-manifest/v1",
        object_sha256=envelope.canonical_sha256(),
    )


def _compile_item_lineage(
    source: BatchQualityItemSource,
    *,
    policy_ref: ObjectRef,
    audit: ContractAudit,
) -> ItemLineageAuditResultV2:
    envelope_ref = _trace_envelope_ref(source.trace_envelope)
    label_refs = tuple(label_decision_ref(value) for value in source.label_decisions)
    selection_ref = selection_context_ref(source.selection_context)
    draft_ref = task_draft_ref(source.task_draft)
    r4_ref = r4_task_contract_set_ref(source.task_contract_set)
    quality_ref = item_quality_compilation_result_ref(source.item_quality)
    base_report_ref = quality_report_v2_ref(source.item_quality.quality_report)
    manifest = source.item_quality.final_package_manifest
    provenance = source.item_quality.provenance_manifest
    environment = source.item_quality.environment_spec
    assert manifest is not None
    assert provenance is not None
    assert environment is not None
    manifest_ref = final_package_manifest_ref(manifest)
    provenance_ref = provenance_manifest_v2_ref(provenance)
    environment_ref = environment_spec_v2_ref(environment)

    trace_failures = sum(
        (
            source.trace_envelope.source_trace_id != source.source_trace_ref.object_id,
            source.trace_envelope.raw_sha256 != source.source_trace_ref.object_sha256,
            source.trace_envelope.adapter_version != source.source_trace_ref.object_version,
            *(value.trace_envelope_ref != envelope_ref for value in source.label_decisions),
        )
    )
    source_trace_check = _relationship_check(
        code=LineageAuditCheckCodeV2.SOURCE_TRACE_LABELS,
        item_id=source.item_id,
        subject_refs=(source.source_trace_ref, envelope_ref),
        evidence_refs=label_refs,
        evaluated_count=3 + len(label_refs),
        failed_count=trace_failures,
        policy_ref=policy_ref,
        audit=audit,
    )
    selection_failures = int(
        source.selection_context.approved_label_decision_refs != _sorted_refs(label_refs)
    )
    selection_check = _relationship_check(
        code=LineageAuditCheckCodeV2.LABEL_SELECTION,
        item_id=source.item_id,
        subject_refs=(selection_ref,),
        evidence_refs=_sorted_refs(
            (
                *label_refs,
                *source.selection_context.approved_label_decision_refs,
            )
        ),
        evaluated_count=max(
            1,
            len(label_refs) + len(source.selection_context.approved_label_decision_refs),
        ),
        failed_count=selection_failures,
        policy_ref=policy_ref,
        audit=audit,
    )
    selection_task_check = _relationship_check(
        code=LineageAuditCheckCodeV2.SELECTION_TASK,
        item_id=source.item_id,
        subject_refs=(draft_ref,),
        evidence_refs=(selection_ref,),
        evaluated_count=1,
        failed_count=int(source.task_draft.selection_context_ref != selection_ref),
        policy_ref=policy_ref,
        audit=audit,
    )
    prompt_gate_ref = source.task_draft.prompt_safety_gate_ref
    assert prompt_gate_ref is not None
    task_r4_check = _relationship_check(
        code=LineageAuditCheckCodeV2.TASK_R4,
        item_id=source.item_id,
        subject_refs=(r4_ref,),
        evidence_refs=(
            draft_ref,
            prompt_gate_ref,
            prompt_leakage_reference_set_ref(source.leakage_reference_set),
        ),
        evaluated_count=3,
        failed_count=sum(
            (
                source.task_contract_set.task_draft_ref != draft_ref,
                source.task_contract_set.task_prompt_safety_gate_ref != prompt_gate_ref,
                source.leakage_reference_set.trace_envelope_ref != envelope_ref,
            )
        ),
        policy_ref=policy_ref,
        audit=audit,
    )
    r4_quality_check = _relationship_check(
        code=LineageAuditCheckCodeV2.R4_ITEM_QUALITY,
        item_id=source.item_id,
        subject_refs=(quality_ref, base_report_ref),
        evidence_refs=(r4_ref,),
        evaluated_count=1,
        failed_count=int(source.item_quality.quality_report.r4_task_contract_set_ref != r4_ref),
        policy_ref=policy_ref,
        audit=audit,
    )
    package_provenance_check = _relationship_check(
        code=LineageAuditCheckCodeV2.PACKAGE_PROVENANCE_SET,
        item_id=source.item_id,
        subject_refs=(manifest_ref, provenance_ref),
        evidence_refs=_sorted_refs(
            (
                *manifest.entry_refs,
                *provenance.package_inventory_entry_refs,
                *provenance.entry_refs,
            )
        )
        or (manifest_ref,),
        evaluated_count=max(1, len(manifest.entry_refs)),
        failed_count=sum(
            (
                provenance.final_package_manifest_ref != manifest_ref,
                provenance.package_inventory_entry_refs != manifest.entry_refs,
                provenance.package_sha256 != manifest.package_sha256,
            )
        ),
        policy_ref=policy_ref,
        audit=audit,
    )
    environment_package_entries = _sorted_refs(
        tuple(ref for artifact in environment.artifacts for ref in artifact.package_inventory_entry_refs)
    )
    package_environment_check = _relationship_check(
        code=LineageAuditCheckCodeV2.PACKAGE_ENVIRONMENT_SET,
        item_id=source.item_id,
        subject_refs=(manifest_ref, provenance_ref, environment_ref),
        evidence_refs=_sorted_refs(
            (
                *environment.artifact_refs,
                *environment.candidate_artifact_version_refs,
                *environment_package_entries,
            )
        )
        or (environment_ref,),
        evaluated_count=max(1, len(environment.artifacts)),
        failed_count=sum(
            (
                environment.final_package_manifest_ref != manifest_ref,
                environment.provenance_manifest_ref != provenance_ref,
                environment.package_sha256 != manifest.package_sha256,
                environment_package_entries != manifest.entry_refs,
            )
        ),
        policy_ref=policy_ref,
        audit=audit,
    )
    member_check = _member_derivation_check(
        source,
        manifest_ref=manifest_ref,
        provenance_ref=provenance_ref,
        policy_ref=policy_ref,
        audit=audit,
    )
    checks_by_code = {
        value.code: value
        for value in (
            source_trace_check,
            selection_check,
            selection_task_check,
            task_r4_check,
            r4_quality_check,
            package_provenance_check,
            package_environment_check,
            member_check,
        )
    }
    checks = tuple(checks_by_code[code] for code in BATCH_QUALITY_ITEM_LINEAGE_CHECKS)
    return ItemLineageAuditResultV2.create(
        item_id=source.item_id,
        source_trace_ref=source.source_trace_ref,
        trace_envelope_ref=envelope_ref,
        label_decision_refs=label_refs,
        selection_context_ref=selection_ref,
        task_draft_ref=draft_ref,
        r4_task_contract_set_ref=r4_ref,
        item_quality_result_ref=quality_ref,
        base_quality_report_ref=base_report_ref,
        final_package_manifest_ref=manifest_ref,
        provenance_manifest_ref=provenance_ref,
        environment_spec_ref=environment_ref,
        checks=checks,
        policy_ref=policy_ref,
        audit=audit,
    )


def _member_derivation_check(
    source: BatchQualityItemSource,
    *,
    manifest_ref: ObjectRef,
    provenance_ref: ObjectRef,
    policy_ref: ObjectRef,
    audit: ContractAudit,
) -> LineageAuditCheckV2:
    manifest = source.item_quality.final_package_manifest
    provenance = source.item_quality.provenance_manifest
    assert manifest is not None
    assert provenance is not None
    if not provenance.entries:
        return LineageAuditCheckV2.create(
            code=LineageAuditCheckCodeV2.PACKAGE_MEMBER_DERIVATION,
            outcome=LineageAuditCheckOutcomeV2.SKIPPED,
            item_ids=(source.item_id,),
            subject_refs=(manifest_ref, provenance_ref),
            evidence_refs=(),
            evaluated_count=0,
            failed_count=0,
            indeterminate_count=0,
            policy_ref=policy_ref,
            audit=audit,
        )
    quality = source.item_quality.quality_report
    failed = 0
    evidence: list[ObjectRef] = []
    for binding in provenance.entries:
        required = {
            *binding.inventory_entry.derivation_root_refs,
            binding.candidate_artifact_version_ref,
            binding.output_ref,
            binding.build_spec_ref,
            binding.artifact_validation_result_ref,
            manifest.candidate_revision_ref,
            quality.deterministic_validation_ref,
            quality.semantic_workflow_result_ref,
        }
        evidence.extend(required)
        if not required.issubset(binding.derivation_closure_refs):
            failed += 1
    return _relationship_check(
        code=LineageAuditCheckCodeV2.PACKAGE_MEMBER_DERIVATION,
        item_id=source.item_id,
        subject_refs=tuple(package_member_provenance_ref(value) for value in provenance.entries),
        evidence_refs=_sorted_refs(tuple(evidence)),
        evaluated_count=len(provenance.entries),
        failed_count=failed,
        policy_ref=policy_ref,
        audit=audit,
    )


def _relationship_check(
    *,
    code: LineageAuditCheckCodeV2,
    item_id: str,
    subject_refs: tuple[ObjectRef, ...],
    evidence_refs: tuple[ObjectRef, ...],
    evaluated_count: int,
    failed_count: int,
    policy_ref: ObjectRef,
    audit: ContractAudit,
) -> LineageAuditCheckV2:
    outcome = LineageAuditCheckOutcomeV2.FAILED if failed_count else LineageAuditCheckOutcomeV2.PASSED
    return LineageAuditCheckV2.create(
        code=code,
        outcome=outcome,
        item_ids=(item_id,),
        subject_refs=subject_refs,
        evidence_refs=evidence_refs,
        evaluated_count=evaluated_count,
        failed_count=failed_count,
        indeterminate_count=0,
        policy_ref=policy_ref,
        audit=audit,
    )


def _compile_batch_alias_check(
    admitted: _AdmittedSources,
    *,
    audit: ContractAudit,
) -> LineageAuditCheckV2:
    owners: dict[ObjectRef, set[str]] = defaultdict(set)
    for source in admitted.sources:
        values = [
            *(label_decision_ref(value) for value in source.label_decisions),
            selection_context_ref(source.selection_context),
            task_draft_ref(source.task_draft),
            r4_task_contract_set_ref(source.task_contract_set),
            item_quality_compilation_result_ref(source.item_quality),
            source.item_quality.quality_report_ref,
        ]
        manifest = source.item_quality.final_package_manifest
        provenance = source.item_quality.provenance_manifest
        environment = source.item_quality.environment_spec
        assert manifest is not None
        assert provenance is not None
        assert environment is not None
        values.extend(
            (
                final_package_manifest_ref(manifest),
                provenance_manifest_v2_ref(provenance),
                environment_spec_v2_ref(environment),
                *environment.artifact_refs,
                *environment.candidate_artifact_version_refs,
                *(value.output_ref for value in environment.artifacts),
            )
        )
        for ref in values:
            owners[ref].add(source.item_id)
    aliases = {ref: item_ids for ref, item_ids in owners.items() if len(item_ids) > 1}
    item_ids = tuple(sorted({item_id for alias_items in aliases.values() for item_id in alias_items}))
    evidence_refs = _sorted_refs(tuple(aliases)) if aliases else _sorted_refs(tuple(owners))
    return LineageAuditCheckV2.create(
        code=LineageAuditCheckCodeV2.CROSS_ITEM_LINEAGE_ALIAS,
        outcome=(LineageAuditCheckOutcomeV2.FAILED if aliases else LineageAuditCheckOutcomeV2.PASSED),
        item_ids=item_ids or tuple(source.item_id for source in admitted.sources),
        subject_refs=(admitted.graph_ref,),
        evidence_refs=evidence_refs,
        evaluated_count=max(1, len(owners)),
        failed_count=len(aliases),
        indeterminate_count=0,
        policy_ref=admitted.policy_ref,
        audit=audit,
    )


def _compile_findings(
    *,
    duplicate_result: DuplicateDetectionResultV2,
    cross_item_result: CrossItemSafetyResultV2,
    lineage_audits: tuple[ItemLineageAuditResultV2, ...],
    batch_lineage_check: LineageAuditCheckV2,
    policy_ref: ObjectRef,
    audit: ContractAudit,
) -> tuple[BatchReviewerFindingV2, ...]:
    findings: list[BatchReviewerFindingV2] = []
    for duplicate_pair in duplicate_result.duplicate_pairs:
        category = {
            (
                DuplicateSubjectKindV2.TASK,
                DuplicateMatchKindV2.EXACT,
            ): BatchFindingCategoryV2.EXACT_TASK_DUPLICATE,
            (
                DuplicateSubjectKindV2.TASK,
                DuplicateMatchKindV2.NEAR,
            ): BatchFindingCategoryV2.NEAR_TASK_DUPLICATE,
            (
                DuplicateSubjectKindV2.ATTACHMENT,
                DuplicateMatchKindV2.EXACT,
            ): BatchFindingCategoryV2.EXACT_ATTACHMENT_DUPLICATE,
            (
                DuplicateSubjectKindV2.ATTACHMENT,
                DuplicateMatchKindV2.NEAR,
            ): BatchFindingCategoryV2.NEAR_ATTACHMENT_DUPLICATE,
        }[(duplicate_pair.subject_kind, duplicate_pair.match_kind)]
        findings.append(
            BatchReviewerFindingV2.create(
                category=category,
                owner_item_id=None,
                affected_item_ids=(
                    duplicate_pair.left_item_id,
                    duplicate_pair.right_item_id,
                ),
                subject_refs=(
                    duplicate_pair.left_subject_ref,
                    duplicate_pair.right_subject_ref,
                ),
                direct_evidence_ref=duplicate_pair_evidence_v2_ref(duplicate_pair),
                restricted_category=None,
                policy_ref=policy_ref,
                audit=audit,
            )
        )
    for visible in cross_item_result.visible_matches:
        findings.append(
            BatchReviewerFindingV2.create(
                category=(
                    BatchFindingCategoryV2.CROSS_ITEM_CONTAMINATION
                    if visible.evidence_kind is CrossItemSafetyEvidenceKindV2.CONTAMINATION
                    else BatchFindingCategoryV2.CROSS_ITEM_LEAKAGE
                ),
                owner_item_id=visible.target_item_id,
                affected_item_ids=(visible.target_item_id,),
                subject_refs=(visible.target_subject_ref,),
                direct_evidence_ref=(cross_item_visible_match_evidence_v2_ref(visible)),
                restricted_category=visible.category,
                policy_ref=policy_ref,
                audit=audit,
            )
        )
    for reuse_pair in cross_item_result.answer_reuse_pairs:
        findings.append(
            _answer_reuse_finding(
                reuse_pair,
                policy_ref=policy_ref,
                audit=audit,
            )
        )
    for lineage in lineage_audits:
        for check in lineage.checks:
            if check.outcome not in {
                LineageAuditCheckOutcomeV2.FAILED,
                LineageAuditCheckOutcomeV2.INDETERMINATE,
            }:
                continue
            findings.append(
                BatchReviewerFindingV2.create(
                    category=(
                        BatchFindingCategoryV2.LINEAGE_INCOMPLETE
                        if check.outcome is LineageAuditCheckOutcomeV2.FAILED
                        else BatchFindingCategoryV2.LINEAGE_INDETERMINATE
                    ),
                    owner_item_id=lineage.item_id,
                    affected_item_ids=(lineage.item_id,),
                    subject_refs=check.subject_refs,
                    direct_evidence_ref=lineage_audit_check_v2_ref(check),
                    restricted_category=None,
                    policy_ref=policy_ref,
                    audit=audit,
                )
            )
    if batch_lineage_check.outcome in {
        LineageAuditCheckOutcomeV2.FAILED,
        LineageAuditCheckOutcomeV2.INDETERMINATE,
    }:
        findings.append(
            BatchReviewerFindingV2.create(
                category=BatchFindingCategoryV2.CROSS_ITEM_LINEAGE_ALIAS,
                owner_item_id=None,
                affected_item_ids=batch_lineage_check.item_ids,
                subject_refs=batch_lineage_check.evidence_refs,
                direct_evidence_ref=lineage_audit_check_v2_ref(batch_lineage_check),
                restricted_category=None,
                policy_ref=policy_ref,
                audit=audit,
            )
        )
    return tuple(
        sorted(
            findings,
            key=lambda value: _ref_key(batch_reviewer_finding_v2_ref(value)),
        )
    )


def _answer_reuse_finding(
    pair: AnswerReusePairEvidenceV2,
    *,
    policy_ref: ObjectRef,
    audit: ContractAudit,
) -> BatchReviewerFindingV2:
    return BatchReviewerFindingV2.create(
        category=BatchFindingCategoryV2.ANSWER_REUSE,
        owner_item_id=None,
        affected_item_ids=(pair.left_item_id, pair.right_item_id),
        subject_refs=(
            pair.left_source_subject_ref,
            pair.right_source_subject_ref,
        ),
        direct_evidence_ref=answer_reuse_pair_evidence_v2_ref(pair),
        restricted_category=None,
        policy_ref=policy_ref,
        audit=audit,
    )


def _compile_revisions(
    admitted: _AdmittedSources,
    *,
    findings: tuple[BatchReviewerFindingV2, ...],
    audit: ContractAudit,
) -> tuple[ItemQualityReportRevisionV2, ...]:
    findings_by_item: dict[str, list[BatchReviewerFindingV2]] = defaultdict(list)
    for finding in findings:
        if finding.scope is BatchFindingScopeV2.ITEM:
            assert finding.owner_item_id is not None
            findings_by_item[finding.owner_item_id].append(finding)
    values: list[ItemQualityReportRevisionV2] = []
    for source in admitted.sources:
        item_findings = tuple(findings_by_item.get(source.item_id, ()))
        if not item_findings:
            continue
        quality = source.item_quality
        assert quality.final_package_manifest_ref is not None
        assert quality.provenance_manifest_ref is not None
        assert quality.environment_spec_ref is not None
        assert quality.package_sha256 is not None
        values.append(
            ItemQualityReportRevisionV2.create(
                item_id=source.item_id,
                base_item_quality_result_ref=(item_quality_compilation_result_ref(quality)),
                base_quality_report_ref=quality.quality_report_ref,
                final_package_manifest_ref=(quality.final_package_manifest_ref),
                provenance_manifest_ref=quality.provenance_manifest_ref,
                environment_spec_ref=quality.environment_spec_ref,
                accepted_artifact_refs=quality.accepted_artifact_refs,
                package_sha256=quality.package_sha256,
                findings=item_findings,
                policy_ref=admitted.policy_ref,
                audit=audit,
            )
        )
    return tuple(sorted(values, key=lambda value: value.item_id))


def _validate_budgets(
    admitted: _AdmittedSources,
    *,
    lineage_audits: tuple[ItemLineageAuditResultV2, ...],
    batch_lineage_check: LineageAuditCheckV2,
    findings: tuple[BatchReviewerFindingV2, ...],
) -> None:
    policy = admitted.policy
    check_count = sum(len(value.checks) for value in lineage_audits) + 1
    if check_count > policy.max_lineage_checks:
        raise BatchQualityPolicyError("batch quality lineage check limit exceeded")
    if len(findings) > policy.max_findings:
        raise BatchQualityPolicyError("batch quality finding budget exceeded")
    evidence_count = (
        sum(len(value.evidence_refs) for lineage in lineage_audits for value in lineage.checks)
        + len(batch_lineage_check.evidence_refs)
        + len(findings)
    )
    if evidence_count > policy.max_direct_evidence_refs:
        raise BatchQualityPolicyError("batch quality direct evidence limit exceeded")
    subject_count = sum(len(value.subject_refs) for value in findings)
    if subject_count > policy.max_finding_subject_refs:
        raise BatchQualityPolicyError("batch quality finding subject limit exceeded")
    item_owner_count = len({value.owner_item_id for value in findings if value.owner_item_id is not None})
    if item_owner_count > policy.max_report_revisions:
        raise BatchQualityPolicyError("batch quality report revision limit exceeded")
    blocked_count = len({item_id for value in findings for item_id in value.blocked_item_ids})
    if blocked_count > policy.max_blocked_items:
        raise BatchQualityPolicyError("batch quality blocked Item limit exceeded")


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _ref_payload(value: ObjectRef) -> dict[str, object]:
    return value.model_dump(mode="json", exclude_none=False)


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    unique = {_ref_key(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


def _require_unique_refs(
    label: str,
    values: tuple[ObjectRef, ...],
) -> None:
    keys = tuple(_ref_key(value) for value in values)
    if len(keys) != len(set(keys)):
        raise BatchQualityPolicyError(f"{label} is duplicated")
