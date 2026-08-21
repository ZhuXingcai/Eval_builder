from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from env_mock_agent.facade import (
    AttachmentDuplicateFingerprintFailureCodeV2,
    AttachmentDuplicateFingerprintOutcomeV2,
    AttachmentDuplicateFingerprintRequestV2,
    AttachmentDuplicateFingerprintResultV2,
    attachment_duplicate_fingerprint_request_ref,
    attachment_duplicate_fingerprint_result_carried_sha256,
)
from eval_factory.batch_quality.duplicates import (
    DuplicateDetectionCompiler,
    DuplicateDetectionItemSource,
    DuplicateDetectionPolicyError,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.duplicate_v2 import (
    DuplicateDetectionPolicyV2,
    DuplicateMatchKindV2,
    DuplicateSubjectKindV2,
)
from eval_factory.contracts.orchestration_v2 import (
    ResolvedJobWorkGraphV2,
    ResolvedWorkUnitV2,
    StageNameV2,
    WorkDependencyJoinModeV2,
    WorkUnitScopeV2,
    dataset_item_id_v2,
    resolved_job_work_graph_v2_ref,
)
from eval_factory.contracts.quality_v2 import (
    EnvironmentArtifactV2,
    EnvironmentSpecV2,
    FinalPackageManifestV2,
    ItemQualityCompilationResultV2,
    ItemQualityOutcomeV2,
    PackageMemberProvenanceV2,
    ProvenanceManifestV2,
    QualityReportV2,
    environment_artifact_ref,
    environment_spec_v2_ref,
    final_package_manifest_ref,
    package_member_provenance_ref,
    provenance_decision_stable_ref,
    provenance_manifest_v2_ref,
    quality_report_v2_ref,
)
from eval_factory.contracts.safety import (
    Disposition,
    OriginClass,
    PackageInventoryMember,
    ProvenanceDecision,
    Visibility,
)
from eval_factory.contracts.task import (
    AttachmentCriticality,
    AttachmentDependency,
    EvidencePriority,
    RequirementConflict,
)
from eval_factory.contracts.task_v2 import (
    R4TaskContractSetV2,
    TaskDraftPromptSafetyStatusV2,
    TaskDraftV2,
    TaskRequirementLineageV2,
    r4_task_contract_set_carried_sha256,
    r4_task_contract_set_ref,
    task_draft_carried_sha256,
    task_draft_ref,
)
from eval_factory.contracts.validation_v2 import (
    CandidatePackageInventoryEntryV2,
    candidate_package_inventory_entry_carried_sha256,
    candidate_package_inventory_entry_ref,
)

HASH = "a" * 64
NOW = datetime(2026, 8, 1, tzinfo=UTC)
GOLD_PATH = (
    Path(__file__).resolve().parents[3]
    / "evals/golden/eval_factory/batch_quality"
    / "r7-01-duplicate-clusters-v1.json"
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
    digest: str | None = None,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r7-01/{suffix}",
        object_version=version,
        object_sha256=digest or _digest(f"{object_type}:{suffix}"),
    )


def _audit(
    *refs: ObjectRef,
    actor: str = "duplicate-compiler-test",
    created_at: datetime = NOW,
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by=actor,
        governing_versions=(
            VersionBinding(
                component="duplicate-detection",
                version="r7-01-v1",
            ),
        ),
        input_refs=tuple(
            sorted(
                set(refs),
                key=lambda ref: (
                    ref.object_type,
                    ref.object_id,
                    ref.object_version,
                    ref.object_sha256,
                ),
            )
        ),
    )


def _evidence(suffix: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=f"evidence-ref://r7-01/{suffix}",
        subject_ref=_ref("input-evidence", suffix),
        source_spans=(
            SourceSpanRef(
                span_id=f"source-span://r7-01/{suffix}",
                source_trace_id=f"trace-source://r7-01/{suffix}",
                raw_sha256=_digest(f"raw:{suffix}"),
            ),
        ),
        polarity=EvidencePolarity.POSITIVE,
        capability="duplicate-detection",
        capability_complete=True,
    )


def _task_draft(
    suffix: str,
    *,
    prompt: str,
    capabilities: tuple[str, ...] = ("workspace-analysis",),
    tools: tuple[str, ...] = ("file-read",),
    attachments: tuple[tuple[str, AttachmentCriticality], ...] = (),
) -> TaskDraftV2:
    episode_ref = _ref("task-episode", suffix)
    dependencies = tuple(
        AttachmentDependency(
            dependency_id=f"attachment-dependency://r7-01/{suffix}/{name}",
            description=f"Input material {name}.",
            criticality=criticality,
            evidence_priority=EvidencePriority.EXPLICIT_REQUIREMENT,
            evidence=(_evidence(f"{suffix}/attachment/{name}"),),
        )
        for name, criticality in attachments
    )
    lineage = TaskRequirementLineageV2(
        requirement_id=f"requirement://r7-01/{suffix}/visible",
        statement="Inspect the supplied input and produce the requested analysis.",
        criticality=AttachmentCriticality.CRITICAL,
        evidence_priority=EvidencePriority.EXPLICIT_REQUIREMENT,
        evidence=(_evidence(f"{suffix}/requirement"),),
        task_episode_refs=(episode_ref,),
        conflict_status=RequirementConflict.NONE,
    )
    pending = TaskDraftV2(
        task_draft_id="task-draft://pending",
        task_version=1,
        supersedes_task_draft_ref=None,
        selection_context_ref=_ref("selection-context", suffix),
        task_episode_refs=(episode_ref,),
        visible_prompt=prompt,
        task_intent="Private authoring intent excluded from duplicate evidence.",
        evaluation_claim="Private evaluation claim excluded from duplicate evidence.",
        required_capabilities=tuple(sorted(capabilities)),
        allowed_tools=tuple(sorted(tools)),
        forbidden_outputs=("original final answer",),
        attachment_dependencies=dependencies,
        requirement_lineage=(lineage,),
        prompt_requirement_ids=(lineage.requirement_id,),
        uncertainties=(),
        prompt_safety_status=TaskDraftPromptSafetyStatusV2.PASSED,
        prompt_safety_gate_ref=_ref("task-prompt-safety-gate", suffix),
        model_profile="internal-task-author-v1",
        prompt_version="task-draft-authoring-prompt/v1",
        policy_version="task-draft-authoring/r4-03-v1",
        task_draft_sha256="0" * 64,
        audit=_audit(),
    )
    digest = task_draft_carried_sha256(pending)
    return pending.model_copy(
        update={
            "task_draft_id": f"task-draft://sha256/{digest}",
            "task_draft_sha256": digest,
        }
    )


def _task_contract_set(
    draft: TaskDraftV2,
    *,
    suffix: str,
) -> R4TaskContractSetV2:
    pending = R4TaskContractSetV2(
        contract_set_id="r4-task-contract-set://pending",
        task_draft_ref=task_draft_ref(draft),
        task_prompt_safety_gate_ref=draft.prompt_safety_gate_ref,
        rubric_set_ref=_ref("rubric-set", suffix),
        evaluator_spec_ref=_ref("evaluator-spec", suffix),
        reference_policy_ref=_ref("reference-policy", suffix),
        tool_policy_ref=_ref("tool-policy", suffix),
        contestant_tool_policy_ref=_ref(
            "contestant-tool-policy",
            suffix,
        ),
        producer_storage_authorization_ref=_ref(
            "producer-storage-authorization",
            suffix,
        ),
        producer_task_view_ref=_ref("producer-task-view", suffix),
        contract_set_sha256="0" * 64,
        audit=_audit(),
    )
    digest = r4_task_contract_set_carried_sha256(pending)
    return pending.model_copy(
        update={
            "contract_set_id": f"r4-task-contract-set://sha256/{digest}",
            "contract_set_sha256": digest,
        }
    )


def _package_entry(
    *,
    suffix: str,
    logical_path: str,
    media_type: str,
    content_sha256: str,
    candidate_ref: ObjectRef,
    output_ref: ObjectRef,
    validation_ref: ObjectRef,
    build_spec_ref: ObjectRef,
) -> CandidatePackageInventoryEntryV2:
    member_ref = _ref(
        "attachment-inventory-member",
        suffix,
        digest=_digest(f"member:{suffix}"),
    )
    member = PackageInventoryMember(
        normalized_path=logical_path,
        member_type="FILE",
        media_type=media_type,
        size_bytes=128,
        content_sha256=content_sha256,
        container_ref=None,
    )
    pending = CandidatePackageInventoryEntryV2(
        inventory_member=member,
        inventory_member_ref=member_ref,
        container_ref=None,
        artifact_build_result_ref=candidate_ref,
        artifact_validation_result_ref=validation_ref,
        output_ref=output_ref,
        build_spec_ref=build_spec_ref,
        derivation_root_refs=(build_spec_ref,),
        entry_sha256="0" * 64,
    )
    digest = candidate_package_inventory_entry_carried_sha256(pending)
    return pending.model_copy(update={"entry_sha256": digest})


def _quality_result(
    *,
    suffix: str,
    contract_set: R4TaskContractSetV2,
    attachments: tuple[
        tuple[str, str, str, str],
        ...,
    ] = (),
) -> ItemQualityCompilationResultV2:
    candidate_revision_ref = _ref("attachment-candidate-revision", suffix)
    deterministic_ref = _ref("revision-deterministic-validation", suffix)
    source_deterministic_ref = _ref(
        "deterministic-item-validation-result",
        suffix,
    )
    semantic_workflow_ref = _ref(
        "semantic-review-workflow-result",
        suffix,
    )
    review_policy_ref = _ref("semantic-review-policy", suffix)
    candidate_inventory_ref = _ref("candidate-package-inventory", suffix) if attachments else None
    candidate_refs: list[ObjectRef] = []
    output_refs: list[ObjectRef] = []
    validation_refs: list[ObjectRef] = []
    entries: list[CandidatePackageInventoryEntryV2] = []
    build_refs: list[ObjectRef] = []
    for artifact_suffix, logical_path, media_type, content_sha256 in attachments:
        candidate_ref = _ref(
            "candidate-artifact-version",
            f"{suffix}/{artifact_suffix}",
        )
        output_ref = _ref(
            "attachment-output",
            f"{suffix}/{artifact_suffix}",
            digest=content_sha256,
        )
        validation_ref = _ref(
            "artifact-deterministic-validation-result",
            f"{suffix}/{artifact_suffix}",
        )
        build_ref = _ref(
            "artifact-build-spec",
            f"{suffix}/{artifact_suffix}",
        )
        candidate_refs.append(candidate_ref)
        output_refs.append(output_ref)
        validation_refs.append(validation_ref)
        build_refs.append(build_ref)
        entries.append(
            _package_entry(
                suffix=f"{suffix}/{artifact_suffix}",
                logical_path=logical_path,
                media_type=media_type,
                content_sha256=content_sha256,
                candidate_ref=candidate_ref,
                output_ref=output_ref,
                validation_ref=validation_ref,
                build_spec_ref=build_ref,
            )
        )
    entry_refs = tuple(candidate_package_inventory_entry_ref(entry) for entry in entries)
    manifest_refs = (
        candidate_revision_ref,
        deterministic_ref,
        source_deterministic_ref,
        *((candidate_inventory_ref,) if candidate_inventory_ref else ()),
        *candidate_refs,
        *output_refs,
        *validation_refs,
        *entry_refs,
    )
    manifest = FinalPackageManifestV2.create(
        candidate_revision_ref=candidate_revision_ref,
        deterministic_validation_ref=deterministic_ref,
        source_deterministic_validation_ref=source_deterministic_ref,
        candidate_inventory_ref=candidate_inventory_ref,
        artifact_version_refs=tuple(candidate_refs),
        output_refs=tuple(output_refs),
        artifact_validation_result_refs=tuple(validation_refs),
        entries=tuple(entries),
        accepted_artifact_refs=tuple(candidate_refs),
        audit=_audit(*manifest_refs),
    )
    manifest_ref = final_package_manifest_ref(manifest)

    provenance_bindings: list[PackageMemberProvenanceV2] = []
    for index, entry in enumerate(entries):
        candidate_ref = candidate_refs[index]
        output_ref = output_refs[index]
        validation_ref = validation_refs[index]
        build_ref = build_refs[index]
        closure = tuple(
            sorted(
                {
                    build_ref,
                    candidate_ref,
                    output_ref,
                    validation_ref,
                    candidate_revision_ref,
                    deterministic_ref,
                    semantic_workflow_ref,
                },
                key=lambda ref: (
                    ref.object_type,
                    ref.object_id,
                    ref.object_version,
                    ref.object_sha256,
                ),
            )
        )
        decision = ProvenanceDecision(
            provenance_decision_id=(f"provenance-decision://r7-01/{suffix}/{index}"),
            subject_ref=entry.inventory_member_ref,
            origin_class=OriginClass.SYSTEM_OR_HARNESS_CONTEXT,
            taint_labels=frozenset(),
            content_risk_labels=frozenset(),
            visibility=Visibility.CONTESTANT_VISIBLE,
            disposition=Disposition.ALLOW_INPUT_EVIDENCE,
            derived_from=closure,
            rule_ids=("item-quality-package/r5-10-v1",),
            source_event_refs=(),
            confidence=1.0,
            review_required=False,
            policy_version="provenance-decision-table/r2-01-v1",
            subject_sha256=entry.inventory_member_ref.object_sha256,
            audit=_audit(entry.inventory_member_ref, *closure),
        )
        decision_ref = provenance_decision_stable_ref(decision)
        binding_refs = (
            manifest_ref,
            candidate_package_inventory_entry_ref(entry),
            entry.inventory_member_ref,
            candidate_ref,
            output_ref,
            build_ref,
            validation_ref,
            *closure,
            decision_ref,
        )
        provenance_bindings.append(
            PackageMemberProvenanceV2.create(
                final_package_manifest_ref=manifest_ref,
                inventory_entry=entry,
                candidate_artifact_version_ref=candidate_ref,
                output_ref=output_ref,
                build_spec_ref=build_ref,
                artifact_validation_result_ref=validation_ref,
                derivation_closure_refs=closure,
                provenance_decision=decision,
                audit=_audit(*binding_refs),
            )
        )
    provenance_entry_refs = tuple(package_member_provenance_ref(item) for item in provenance_bindings)
    provenance = ProvenanceManifestV2.create(
        final_package_manifest_ref=manifest_ref,
        candidate_revision_ref=candidate_revision_ref,
        package_sha256=manifest.package_sha256,
        package_inventory_entry_refs=entry_refs,
        entries=tuple(provenance_bindings),
        audit=_audit(
            manifest_ref,
            candidate_revision_ref,
            *entry_refs,
            *provenance_entry_refs,
        ),
    )
    provenance_ref = provenance_manifest_v2_ref(provenance)

    environment_artifacts = tuple(
        EnvironmentArtifactV2.create(
            artifact_id=f"artifact://r7-01/{suffix}/{artifact_suffix}",
            attachment_dependency_id=(f"attachment-dependency://r7-01/{suffix}/{artifact_suffix}"),
            candidate_artifact_version_ref=candidate_refs[index],
            output_ref=output_refs[index],
            logical_path=logical_path,
            media_type=media_type,
            size_bytes=128,
            artifact_validation_result_ref=validation_refs[index],
            package_inventory_entry_refs=(candidate_package_inventory_entry_ref(entries[index]),),
        )
        for index, (
            artifact_suffix,
            logical_path,
            media_type,
            _,
        ) in enumerate(attachments)
    )
    environment_refs = tuple(environment_artifact_ref(item) for item in environment_artifacts)
    environment = EnvironmentSpecV2.create(
        candidate_revision_ref=candidate_revision_ref,
        final_package_manifest_ref=manifest_ref,
        provenance_manifest_ref=provenance_ref,
        candidate_artifact_version_refs=tuple(candidate_refs),
        artifacts=environment_artifacts,
        package_sha256=manifest.package_sha256,
        audit=_audit(
            candidate_revision_ref,
            manifest_ref,
            provenance_ref,
            *candidate_refs,
            *environment_refs,
        ),
    )
    environment_ref = environment_spec_v2_ref(environment)

    semantic_round_refs = tuple(
        _ref("semantic-review-round-result", f"{suffix}/{index}") for index in range(3)
    )
    stage_result_refs = tuple(
        _ref(
            "stage-result",
            f"{suffix}/{index}",
            version="record/v1",
        )
        for index in range(3)
    )
    report_refs = (
        r4_task_contract_set_ref(contract_set),
        review_policy_ref,
        semantic_workflow_ref,
        candidate_revision_ref,
        deterministic_ref,
        source_deterministic_ref,
        *((candidate_inventory_ref,) if candidate_inventory_ref else ()),
        *candidate_refs,
        *output_refs,
        *validation_refs,
        *semantic_round_refs,
        *stage_result_refs,
        *candidate_refs,
        manifest_ref,
        provenance_ref,
        environment_ref,
    )
    report = QualityReportV2.create(
        r4_task_contract_set_ref=r4_task_contract_set_ref(contract_set),
        review_policy_ref=review_policy_ref,
        semantic_workflow_result_ref=semantic_workflow_ref,
        candidate_revision_ref=candidate_revision_ref,
        deterministic_validation_ref=deterministic_ref,
        source_deterministic_validation_ref=source_deterministic_ref,
        candidate_inventory_ref=candidate_inventory_ref,
        artifact_version_refs=tuple(candidate_refs),
        output_refs=tuple(output_refs),
        artifact_validation_result_refs=tuple(validation_refs),
        semantic_round_result_refs=semantic_round_refs,
        stage_result_refs=stage_result_refs,
        repair_plan_refs=(),
        repair_result_refs=(),
        current_finding_refs=(),
        stale_finding_refs=(),
        resolution_refs=(),
        accepted_artifact_refs=tuple(candidate_refs),
        final_package_manifest_ref=manifest_ref,
        provenance_manifest_ref=provenance_ref,
        environment_spec_ref=environment_ref,
        package_sha256=manifest.package_sha256,
        input_state_only=True,
        outcome=ItemQualityOutcomeV2.PASSED,
        failure_code=None,
        open_p0_count=0,
        open_p1_count=0,
        unresolved_non_waivable_count=0,
        approvable=True,
        audit=_audit(*report_refs),
    )
    report_ref = quality_report_v2_ref(report)
    return ItemQualityCompilationResultV2.create(
        quality_report=report,
        final_package_manifest=manifest,
        provenance_manifest=provenance,
        environment_spec=environment,
        audit=_audit(
            report_ref,
            manifest_ref,
            provenance_ref,
            environment_ref,
            *candidate_refs,
        ),
    )


def _graph(count: int) -> tuple[ResolvedJobWorkGraphV2, tuple[str, ...]]:
    job_id = "dataset-job://r7-01/compiler"
    source_refs = tuple(
        _ref("trace-source", f"source/{index}", version="raw_traj_v1") for index in range(count)
    )
    item_ids = tuple(dataset_item_id_v2(job_id, source_ref) for source_ref in source_refs)
    work_units = tuple(
        ResolvedWorkUnitV2.create(
            scope=WorkUnitScopeV2.ITEM,
            stage=StageNameV2.TRACE_INDEX,
            job_id=job_id,
            item_id=item_id,
            source_trace_ref=source_ref,
            artifact_execution_group_ref=None,
            depends_on_work_unit_refs=(),
            join_mode=WorkDependencyJoinModeV2.ALL_SUCCEEDED,
        )
        for item_id, source_ref in zip(
            item_ids,
            source_refs,
            strict=True,
        )
    )
    job_ref = ObjectRef(
        object_type="dataset-job-spec",
        object_id=job_id,
        object_version="v2",
        object_sha256=_digest("dataset-job-spec"),
    )
    plan_ref = _ref("resolved-dataset-job-plan", "compiler")
    graph = ResolvedJobWorkGraphV2.create(
        job_id=job_id,
        dataset_job_spec_ref=job_ref,
        resolved_job_plan_ref=plan_ref,
        item_ids=item_ids,
        source_trace_refs=source_refs,
        work_units=work_units,
        audit=_audit(job_ref, plan_ref),
    )
    return graph, item_ids


def _source(
    *,
    item_id: str,
    suffix: str,
    prompt: str,
    attachments: tuple[
        tuple[str, str, str, str],
        ...,
    ] = (),
    capabilities: tuple[str, ...] = ("workspace-analysis",),
) -> DuplicateDetectionItemSource:
    draft = _task_draft(
        suffix,
        prompt=prompt,
        capabilities=capabilities,
        attachments=tuple(
            (
                artifact_suffix,
                AttachmentCriticality.REQUIRED,
            )
            for artifact_suffix, _, _, _ in attachments
        ),
    )
    contract_set = _task_contract_set(draft, suffix=suffix)
    return DuplicateDetectionItemSource(
        item_id=item_id,
        task_draft=draft,
        task_contract_set=contract_set,
        item_quality=_quality_result(
            suffix=suffix,
            contract_set=contract_set,
            attachments=attachments,
        ),
    )


def _policy(
    *,
    task_threshold: int = 9_000,
    attachment_threshold: int = 9_000,
    max_pair_comparisons: int = 1_000,
) -> DuplicateDetectionPolicyV2:
    from env_mock_agent.facade import (
        AttachmentDuplicateFingerprintLimitsV2,
    )

    limits = AttachmentDuplicateFingerprintLimitsV2(
        max_file_bytes=10_000_000,
        max_extracted_characters=1_000_000,
        max_inventory_members=10_000,
        max_nested_depth=4,
        max_expanded_bytes=50_000_000,
        max_compression_ratio_milli=1_000_000,
        min_token_count=4,
        shingle_size=3,
        fingerprint_bits=256,
    )
    return DuplicateDetectionPolicyV2.create(
        task_near_threshold_bps=task_threshold,
        attachment_near_threshold_bps=attachment_threshold,
        task_shingle_size=3,
        attachment_shingle_size=3,
        fingerprint_bits=256,
        min_task_tokens=4,
        min_attachment_tokens=4,
        max_task_characters=100_000,
        max_attachment_characters=1_000_000,
        max_items=100,
        max_attachments=1_000,
        max_pair_comparisons=max_pair_comparisons,
        attachment_fingerprint_limits=limits,
        audit=_audit(),
    )


class _FingerprintFacade:
    def __init__(
        self,
        *,
        fingerprints: dict[str, str | None] | None = None,
        blocked_subjects: set[str] | None = None,
    ) -> None:
        self.fingerprints = fingerprints or {}
        self.blocked_subjects = blocked_subjects or set()
        self.calls: list[str] = []

    async def fingerprint(
        self,
        request: AttachmentDuplicateFingerprintRequestV2,
    ) -> AttachmentDuplicateFingerprintResultV2:
        subject_id = request.environment_artifact_ref.object_id
        self.calls.append(subject_id)
        if subject_id in self.blocked_subjects:
            outcome = AttachmentDuplicateFingerprintOutcomeV2.BLOCKED
            fingerprint = None
            token_count = 0
            shingle_count = 0
            failure_code = AttachmentDuplicateFingerprintFailureCodeV2.OUTPUT_HASH_MISMATCH
            extractor_version = None
        else:
            configured = self.fingerprints.get(
                subject_id,
                _digest(request.content_sha256),
            )
            if configured is None:
                outcome = AttachmentDuplicateFingerprintOutcomeV2.UNSUPPORTED
                fingerprint = None
                token_count = 0
                shingle_count = 0
                failure_code = AttachmentDuplicateFingerprintFailureCodeV2.MEDIA_UNSUPPORTED
                extractor_version = None
            else:
                outcome = AttachmentDuplicateFingerprintOutcomeV2.SUPPORTED
                fingerprint = configured
                token_count = 12
                shingle_count = 10
                failure_code = None
                extractor_version = "attachment-duplicate-extractor/r7-01-v1"
        pending = AttachmentDuplicateFingerprintResultV2(
            fingerprint_result_id=("attachment-duplicate-fingerprint-result://pending"),
            fingerprint_request_ref=(attachment_duplicate_fingerprint_request_ref(request)),
            environment_artifact_ref=request.environment_artifact_ref,
            output_ref=request.output_ref,
            exact_content_sha256=request.content_sha256,
            outcome=outcome,
            similarity_fingerprint=fingerprint,
            token_count=token_count,
            shingle_count=shingle_count,
            failure_code=failure_code,
            extractor_version=extractor_version,
            normalization_version=request.normalization_version,
            policy_version=request.policy_version,
            result_sha256="0" * 64,
        )
        digest = attachment_duplicate_fingerprint_result_carried_sha256(pending)
        return pending.model_copy(
            update={
                "fingerprint_result_id": (f"attachment-duplicate-fingerprint-result://sha256/{digest}"),
                "result_sha256": digest,
            }
        )


@pytest.mark.asyncio
async def test_exact_task_duplicates_form_content_free_cluster() -> None:
    graph, item_ids = _graph(3)
    sources = (
        _source(
            item_id=item_ids[0],
            suffix="left",
            prompt="Inspect the supplied workspace and summarize its design.",
        ),
        _source(
            item_id=item_ids[1],
            suffix="right",
            prompt="Inspect the supplied workspace and summarize its design.",
        ),
        _source(
            item_id=item_ids[2],
            suffix="unrelated",
            prompt="Calculate monthly invoice totals from the ledger.",
        ),
    )
    facade = _FingerprintFacade()
    compiler = DuplicateDetectionCompiler()

    result = await compiler.compile(
        resolved_job_work_graph=graph,
        policy=_policy(task_threshold=10_000),
        sources=sources,
        facade=facade,
        audit=_audit(),
    )

    task_pairs = tuple(
        pair for pair in result.duplicate_pairs if pair.subject_kind is DuplicateSubjectKindV2.TASK
    )
    assert len(task_pairs) == 1
    assert task_pairs[0].match_kind is DuplicateMatchKindV2.EXACT
    assert len(result.duplicate_clusters) == 1
    assert result.duplicate_clusters[0].member_item_ids == item_ids[:2]
    assert facade.calls == []
    serialized = json.dumps(result.model_dump(mode="json"), sort_keys=True)
    assert "summarize its design" not in serialized
    compiler.validate_current(
        result,
        resolved_job_work_graph=graph,
        policy=_policy(task_threshold=10_000),
        sources=sources,
    )


@pytest.mark.asyncio
async def test_task_public_shape_changes_exact_identity() -> None:
    graph, item_ids = _graph(2)
    prompt = "Inspect the supplied workspace and summarize its design."
    sources = (
        _source(
            item_id=item_ids[0],
            suffix="left",
            prompt=prompt,
            capabilities=("workspace-analysis",),
        ),
        _source(
            item_id=item_ids[1],
            suffix="right",
            prompt=prompt,
            capabilities=("workspace-analysis", "spreadsheet-analysis"),
        ),
    )

    result = await DuplicateDetectionCompiler().compile(
        resolved_job_work_graph=graph,
        policy=_policy(task_threshold=10_000),
        sources=sources,
        facade=_FingerprintFacade(),
        audit=_audit(),
    )

    assert result.task_fingerprints[0].exact_task_sha256 != result.task_fingerprints[1].exact_task_sha256
    assert not result.duplicate_pairs


@pytest.mark.asyncio
async def test_attachment_exact_near_and_unsupported_evidence() -> None:
    graph, item_ids = _graph(4)
    shared_hash = _digest("shared attachment")
    near_hash = _digest("near attachment")
    binary_hash = _digest("binary attachment")
    attachments = (
        (("input", "inputs/input.txt", "text/plain", shared_hash),),
        (("input", "inputs/input.txt", "text/plain", shared_hash),),
        (("input", "inputs/input.txt", "text/plain", near_hash),),
        (
            (
                "input",
                "inputs/input.bin",
                "application/octet-stream",
                binary_hash,
            ),
        ),
    )
    sources = tuple(
        _source(
            item_id=item_ids[index],
            suffix=f"item-{index}",
            prompt=f"Perform distinct analysis number {index}.",
            attachments=attachments[index],
        )
        for index in range(4)
    )
    subject_ids = tuple(
        source.item_quality.environment_spec.artifacts[0].artifact_sha256 for source in sources
    )
    environment_ids = tuple(f"environment-artifact://sha256/{digest}" for digest in subject_ids)
    facade = _FingerprintFacade(
        fingerprints={
            environment_ids[0]: "0" * 64,
            environment_ids[1]: "0" * 64,
            environment_ids[2]: "0" * 64,
            environment_ids[3]: None,
        }
    )
    compiler = DuplicateDetectionCompiler()
    policy = _policy(
        task_threshold=10_000,
        attachment_threshold=10_000,
    )

    result = await compiler.compile(
        resolved_job_work_graph=graph,
        policy=policy,
        sources=sources,
        facade=facade,
        audit=_audit(),
    )

    attachment_pairs = tuple(
        pair for pair in result.duplicate_pairs if pair.subject_kind is DuplicateSubjectKindV2.ATTACHMENT
    )
    assert tuple(pair.match_kind for pair in attachment_pairs).count(DuplicateMatchKindV2.EXACT) == 1
    assert tuple(pair.match_kind for pair in attachment_pairs).count(DuplicateMatchKindV2.NEAR) == 2
    assert result.evaluated_attachment_pair_count == 6
    assert result.unsupported_attachment_count == 1
    attachment_clusters = tuple(
        cluster
        for cluster in result.duplicate_clusters
        if cluster.subject_kind is DuplicateSubjectKindV2.ATTACHMENT
    )
    assert len(attachment_clusters) == 1
    assert len(attachment_clusters[0].member_subject_refs) == 3
    assert len(facade.calls) == 4
    compiler.validate_current(
        result,
        resolved_job_work_graph=graph,
        policy=policy,
        sources=sources,
    )
    assert len(facade.calls) == 4


@pytest.mark.asyncio
async def test_connected_component_does_not_fabricate_transitive_pair() -> None:
    graph, item_ids = _graph(3)
    sources = tuple(
        _source(
            item_id=item_ids[index],
            suffix=f"chain-{index}",
            prompt=f"Perform distinct chain analysis {index}.",
            attachments=(
                (
                    "input",
                    "inputs/input.txt",
                    "text/plain",
                    _digest(f"chain attachment {index}"),
                ),
            ),
        )
        for index in range(3)
    )
    environment_ids = tuple(
        environment_artifact_ref(source.item_quality.environment_spec.artifacts[0]).object_id
        for source in sources
    )
    facade = _FingerprintFacade(
        fingerprints={
            environment_ids[0]: "0" * 64,
            environment_ids[1]: "fffff" + "0" * 59,
            environment_ids[2]: "f" * 10 + "0" * 54,
        }
    )

    result = await DuplicateDetectionCompiler().compile(
        resolved_job_work_graph=graph,
        policy=_policy(
            task_threshold=10_000,
            attachment_threshold=9_000,
        ),
        sources=sources,
        facade=facade,
        audit=_audit(),
    )

    attachment_pairs = tuple(
        pair for pair in result.duplicate_pairs if pair.subject_kind is DuplicateSubjectKindV2.ATTACHMENT
    )
    assert len(attachment_pairs) == 2
    assert {frozenset((pair.left_item_id, pair.right_item_id)) for pair in attachment_pairs} == {
        frozenset((item_ids[0], item_ids[1])),
        frozenset((item_ids[1], item_ids[2])),
    }
    cluster = next(
        cluster
        for cluster in result.duplicate_clusters
        if cluster.subject_kind is DuplicateSubjectKindV2.ATTACHMENT
    )
    assert len(cluster.member_subject_refs) == 3
    assert len(cluster.direct_pair_refs) == 2


@pytest.mark.asyncio
async def test_input_order_and_audit_do_not_change_result_identity() -> None:
    graph, item_ids = _graph(2)
    sources = (
        _source(
            item_id=item_ids[0],
            suffix="stable-left",
            prompt="Inspect the supplied workspace and summarize its design.",
        ),
        _source(
            item_id=item_ids[1],
            suffix="stable-right",
            prompt="Inspect the supplied workspace and summarize its design.",
        ),
    )
    policy = _policy(task_threshold=10_000)
    compiler = DuplicateDetectionCompiler()

    first = await compiler.compile(
        resolved_job_work_graph=graph,
        policy=policy,
        sources=sources,
        facade=_FingerprintFacade(),
        audit=_audit(),
    )
    second = await compiler.compile(
        resolved_job_work_graph=graph,
        policy=policy,
        sources=tuple(reversed(sources)),
        facade=_FingerprintFacade(),
        audit=_audit(
            actor="other-compiler",
            created_at=datetime(2026, 8, 2, tzinfo=UTC),
        ),
    )

    assert duplicate_result_identity(first) == duplicate_result_identity(second)


def duplicate_result_identity(result) -> tuple[str, str]:
    return result.duplicate_detection_result_id, result.result_sha256


@pytest.mark.asyncio
async def test_invalid_sources_budget_and_blocked_facade_fail_closed() -> None:
    graph, item_ids = _graph(2)
    sources = (
        _source(
            item_id=item_ids[0],
            suffix="invalid-left",
            prompt="Inspect the supplied workspace and summarize its design.",
        ),
        _source(
            item_id=item_ids[1],
            suffix="invalid-right",
            prompt="Inspect a different workspace and explain its design.",
        ),
    )
    facade = _FingerprintFacade()
    compiler = DuplicateDetectionCompiler()

    with pytest.raises(DuplicateDetectionPolicyError, match="pair budget"):
        await compiler.compile(
            resolved_job_work_graph=graph,
            policy=_policy(max_pair_comparisons=0),
            sources=sources,
            facade=facade,
            audit=_audit(),
        )
    assert facade.calls == []

    with pytest.raises(DuplicateDetectionPolicyError, match="duplicate item"):
        await compiler.compile(
            resolved_job_work_graph=graph,
            policy=_policy(),
            sources=(sources[0], replace(sources[0])),
            facade=facade,
            audit=_audit(),
        )

    shared_subject = replace(
        sources[0],
        item_id=item_ids[1],
    )
    with pytest.raises(
        DuplicateDetectionPolicyError,
        match="task subject ownership",
    ):
        await compiler.compile(
            resolved_job_work_graph=graph,
            policy=_policy(),
            sources=(sources[0], shared_subject),
            facade=facade,
            audit=_audit(),
        )

    stale = replace(
        sources[0],
        task_draft=sources[0].task_draft.model_copy(
            update={"visible_prompt": "Mutated without a new identity."}
        ),
    )
    with pytest.raises(DuplicateDetectionPolicyError, match="TaskDraft"):
        await compiler.compile(
            resolved_job_work_graph=graph,
            policy=_policy(),
            sources=(stale,),
            facade=facade,
            audit=_audit(),
        )

    attached = _source(
        item_id=item_ids[0],
        suffix="blocked",
        prompt="Inspect the supplied binary input.",
        attachments=(
            (
                "input",
                "inputs/input.bin",
                "application/octet-stream",
                _digest("blocked attachment"),
            ),
        ),
    )
    blocked_subject = environment_artifact_ref(attached.item_quality.environment_spec.artifacts[0]).object_id
    blocked_facade = _FingerprintFacade(blocked_subjects={blocked_subject})
    with pytest.raises(DuplicateDetectionPolicyError, match="blocked"):
        await compiler.compile(
            resolved_job_work_graph=graph,
            policy=_policy(),
            sources=(attached,),
            facade=blocked_facade,
            audit=_audit(),
        )
    assert blocked_facade.calls == [blocked_subject]


def test_graph_ref_is_current_fixture_authority() -> None:
    graph, _ = _graph(1)
    assert resolved_job_work_graph_v2_ref(graph).object_sha256 == (graph.resolved_job_work_graph_sha256)


def test_duplicate_cluster_gold_is_complete_and_content_free() -> None:
    payload = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    assert payload["policy_version"] == "duplicate-detection/r7-01-v1"
    assert payload["clustering_mode"] == "CONNECTED_COMPONENTS"
    assert payload["direct_pair_authority"] is True
    assert payload["stable_hash_seeds"] == [1, 321]
    assert {scenario["scenario_id"] for scenario in payload["scenarios"]} == {
        "exact-task",
        "near-task",
        "unrelated",
        "connected-transitive",
        "multi-cluster",
        "no-attachment",
        "unsupported-attachment",
    }
    transitive = next(
        scenario for scenario in payload["scenarios"] if scenario["scenario_id"] == "connected-transitive"
    )
    assert transitive["cluster_member_counts"] == [3]
    assert transitive["direct_pair_count"] == 2
    assert transitive["transitive_unobserved_pair_count"] == 1
    serialized = json.dumps(payload, sort_keys=True).casefold()
    for forbidden in (
        "raw_trace",
        "visible_prompt",
        "extracted_text",
        "token_inventory",
        "shingle_inventory",
        "physical_path",
        "private_reference",
        "final_answer",
        "grader_rule",
        "hidden_condition",
        "release_decision",
        "batch_quality_report",
    ):
        assert forbidden not in serialized
