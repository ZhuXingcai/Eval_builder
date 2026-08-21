from __future__ import annotations

import hashlib
from dataclasses import dataclass

from eval_factory.attachment_planning.review import (
    IsolatedSemanticReviewOrchestrator,
    SemanticReviewPolicyError,
)
from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.quality import Severity
from eval_factory.contracts.quality_v2 import (
    ITEM_QUALITY_POLICY_VERSION,
    EnvironmentArtifactV2,
    EnvironmentSpecV2,
    FinalPackageManifestV2,
    ItemQualityCompilationResultV2,
    ItemQualityFailureCodeV2,
    ItemQualityOutcomeV2,
    PackageMemberProvenanceV2,
    ProvenanceManifestV2,
    QualityReportV2,
    environment_spec_v2_ref,
    final_package_manifest_ref,
    package_member_provenance_ref,
    provenance_decision_stable_ref,
    provenance_manifest_v2_ref,
    quality_report_v2_ref,
)
from eval_factory.contracts.review_v2 import (
    AttachmentCandidateRevisionV2,
    CandidateArtifactVersionV2,
    RevisionDeterministicValidationOutcomeV2,
    RevisionDeterministicValidationV2,
    SemanticReviewFindingV2,
    SemanticReviewPolicyV2,
    SemanticReviewWorkflowResultV2,
    attachment_candidate_revision_carried_sha256,
    attachment_candidate_revision_ref,
    candidate_artifact_version_carried_sha256,
    candidate_artifact_version_ref,
    revision_deterministic_validation_carried_sha256,
    revision_deterministic_validation_ref,
    semantic_review_finding_ref,
    semantic_review_policy_carried_sha256,
    semantic_review_policy_ref,
    semantic_review_workflow_result_carried_sha256,
    semantic_review_workflow_result_ref,
)
from eval_factory.contracts.safety import (
    Disposition,
    OriginClass,
    Visibility,
)
from eval_factory.contracts.task_v2 import (
    R4TaskContractSetV2,
    r4_task_contract_set_carried_sha256,
    r4_task_contract_set_ref,
)
from eval_factory.contracts.validation_v2 import (
    ArtifactDeterministicValidationOutcomeV2,
    ArtifactDeterministicValidationResultV2,
    CandidatePackageInventoryEntryV2,
    DeterministicItemValidationOutcomeV2,
    DeterministicItemValidationResultV2,
    DeterministicValidationFindingV2,
    candidate_package_inventory_entry_ref,
    candidate_package_inventory_ref,
    deterministic_item_validation_result_ref,
    deterministic_validation_finding_ref,
    validate_deterministic_item_validation_result_identity,
)
from eval_factory.orchestration import JobStore
from eval_factory.provenance.decisions import (
    ProvenanceDecisionInput,
    ProvenanceDecisionTable,
)


class ItemQualityPolicyError(RuntimeError):
    pass


class _PackageBlocked(RuntimeError):
    def __init__(self, code: ItemQualityFailureCodeV2) -> None:
        super().__init__(code.value)
        self.code = code


@dataclass(frozen=True, slots=True)
class _ArtifactContext:
    artifact: CandidateArtifactVersionV2
    validation: ArtifactDeterministicValidationResultV2
    validation_ref: ObjectRef


@dataclass(frozen=True, slots=True)
class _FinalPackageValues:
    manifest: FinalPackageManifestV2
    provenance: ProvenanceManifestV2
    environment: EnvironmentSpecV2


class ItemQualityCompiler:
    def compile(
        self,
        *,
        task_contract_set: R4TaskContractSetV2,
        review_policy: SemanticReviewPolicyV2,
        semantic_workflow: SemanticReviewWorkflowResultV2,
        candidate_revision: AttachmentCandidateRevisionV2,
        deterministic_validation: RevisionDeterministicValidationV2,
        source_deterministic_validation: DeterministicItemValidationResultV2,
        job_store: JobStore,
        job_id: Identifier,
        item_id: Identifier,
        audit: ContractAudit,
    ) -> ItemQualityCompilationResultV2:
        try:
            self._validate_sources(
                task_contract_set=task_contract_set,
                review_policy=review_policy,
                semantic_workflow=semantic_workflow,
                candidate_revision=candidate_revision,
                deterministic_validation=deterministic_validation,
                source_deterministic_validation=source_deterministic_validation,
                job_store=job_store,
                job_id=job_id,
                item_id=item_id,
            )
            counts = _finding_counts(
                semantic_workflow,
                source_deterministic_validation,
            )
            workflow_outcome = ItemQualityOutcomeV2(semantic_workflow.outcome.value)
            if workflow_outcome is not ItemQualityOutcomeV2.PASSED:
                return self._result(
                    task_contract_set=task_contract_set,
                    review_policy=review_policy,
                    semantic_workflow=semantic_workflow,
                    candidate_revision=candidate_revision,
                    deterministic_validation=deterministic_validation,
                    source_deterministic_validation=(source_deterministic_validation),
                    outcome=workflow_outcome,
                    failure_code=None,
                    counts=counts,
                    final_values=None,
                    audit=audit,
                )
            _validate_passing_quality(
                semantic_workflow=semantic_workflow,
                deterministic_validation=deterministic_validation,
                source_deterministic_validation=(source_deterministic_validation),
                counts=counts,
            )
            try:
                final_values = self._finalize_package(
                    semantic_workflow=semantic_workflow,
                    candidate_revision=candidate_revision,
                    deterministic_validation=deterministic_validation,
                    source_deterministic_validation=(source_deterministic_validation),
                    audit=audit,
                )
            except _PackageBlocked as exc:
                return self._result(
                    task_contract_set=task_contract_set,
                    review_policy=review_policy,
                    semantic_workflow=semantic_workflow,
                    candidate_revision=candidate_revision,
                    deterministic_validation=deterministic_validation,
                    source_deterministic_validation=(source_deterministic_validation),
                    outcome=ItemQualityOutcomeV2.BLOCKED,
                    failure_code=exc.code,
                    counts=counts,
                    final_values=None,
                    audit=audit,
                )
            return self._result(
                task_contract_set=task_contract_set,
                review_policy=review_policy,
                semantic_workflow=semantic_workflow,
                candidate_revision=candidate_revision,
                deterministic_validation=deterministic_validation,
                source_deterministic_validation=(source_deterministic_validation),
                outcome=ItemQualityOutcomeV2.PASSED,
                failure_code=None,
                counts=counts,
                final_values=final_values,
                audit=audit,
            )
        except ItemQualityPolicyError:
            raise
        except (SemanticReviewPolicyError, ValueError) as exc:
            raise ItemQualityPolicyError(f"item quality compilation failed: {exc}") from exc

    def validate_current(
        self,
        result: ItemQualityCompilationResultV2,
        *,
        task_contract_set: R4TaskContractSetV2,
        review_policy: SemanticReviewPolicyV2,
        semantic_workflow: SemanticReviewWorkflowResultV2,
        candidate_revision: AttachmentCandidateRevisionV2,
        deterministic_validation: RevisionDeterministicValidationV2,
        source_deterministic_validation: DeterministicItemValidationResultV2,
        job_store: JobStore,
        job_id: Identifier,
        item_id: Identifier,
    ) -> None:
        try:
            canonical = ItemQualityCompilationResultV2.model_validate(result.model_dump(mode="python"))
            if canonical != result:
                raise ItemQualityPolicyError("item quality canonical payload changed")
            rebuilt = self.compile(
                task_contract_set=task_contract_set,
                review_policy=review_policy,
                semantic_workflow=semantic_workflow,
                candidate_revision=candidate_revision,
                deterministic_validation=deterministic_validation,
                source_deterministic_validation=(source_deterministic_validation),
                job_store=job_store,
                job_id=job_id,
                item_id=item_id,
                audit=result.audit,
            )
            if _without_audit_actor_time(rebuilt) != _without_audit_actor_time(result):
                raise ItemQualityPolicyError("item quality result is stale")
        except ItemQualityPolicyError:
            raise
        except ValueError as exc:
            raise ItemQualityPolicyError(f"current item quality validation failed: {exc}") from exc

    @staticmethod
    def _validate_sources(
        *,
        task_contract_set: R4TaskContractSetV2,
        review_policy: SemanticReviewPolicyV2,
        semantic_workflow: SemanticReviewWorkflowResultV2,
        candidate_revision: AttachmentCandidateRevisionV2,
        deterministic_validation: RevisionDeterministicValidationV2,
        source_deterministic_validation: DeterministicItemValidationResultV2,
        job_store: JobStore,
        job_id: Identifier,
        item_id: Identifier,
    ) -> None:
        for value, model_type, label in (
            (
                task_contract_set,
                R4TaskContractSetV2,
                "R4 task contract set",
            ),
            (
                review_policy,
                SemanticReviewPolicyV2,
                "semantic review policy",
            ),
            (
                candidate_revision,
                AttachmentCandidateRevisionV2,
                "candidate revision",
            ),
            (
                deterministic_validation,
                RevisionDeterministicValidationV2,
                "revision deterministic validation",
            ),
            (
                source_deterministic_validation,
                DeterministicItemValidationResultV2,
                "source deterministic validation",
            ),
        ):
            canonical = model_type.model_validate(value.model_dump(mode="python"))
            if canonical != value:
                raise ItemQualityPolicyError(f"{label} canonical payload changed")
        if task_contract_set.contract_set_sha256 != r4_task_contract_set_carried_sha256(task_contract_set):
            raise ItemQualityPolicyError("R4 task contract set identity is stale")
        if review_policy.policy_sha256 != semantic_review_policy_carried_sha256(review_policy):
            raise ItemQualityPolicyError("semantic review policy identity is stale")
        if candidate_revision.candidate_revision_sha256 != attachment_candidate_revision_carried_sha256(
            candidate_revision
        ) or any(
            item.artifact_version_sha256 != candidate_artifact_version_carried_sha256(item)
            for item in candidate_revision.artifact_versions
        ):
            raise ItemQualityPolicyError("candidate revision identity is stale")
        if deterministic_validation.validation_sha256 != revision_deterministic_validation_carried_sha256(
            deterministic_validation
        ):
            raise ItemQualityPolicyError("revision deterministic validation identity is stale")
        validate_deterministic_item_validation_result_identity(source_deterministic_validation)
        source_ref = deterministic_item_validation_result_ref(source_deterministic_validation)
        if (
            deterministic_validation.candidate_revision_ref
            != attachment_candidate_revision_ref(candidate_revision)
            or deterministic_validation.source_deterministic_validation_result_ref != source_ref
            or deterministic_validation.artifact_validation_result_refs
            != _sorted_refs(source_deterministic_validation.artifact_validation_result_refs)
            or deterministic_validation.finding_refs != source_deterministic_validation.finding_refs
            or deterministic_validation.output_refs != candidate_revision.candidate_output_refs
        ):
            raise ItemQualityPolicyError("current deterministic validation source is stale")
        expected_source_subject = (
            candidate_revision.base_reconstruction_result_ref
            if candidate_revision.revision == 1
            else attachment_candidate_revision_ref(candidate_revision)
        )
        if source_deterministic_validation.attachment_reconstruction_result_ref != expected_source_subject:
            raise ItemQualityPolicyError("source deterministic validation belongs to another candidate")
        if task_contract_set.producer_task_view_ref != source_deterministic_validation.producer_task_view_ref:
            raise ItemQualityPolicyError("producer task view does not match current quality sources")
        inventory = source_deterministic_validation.candidate_inventory
        if inventory is not None and (
            inventory.attachment_reconstruction_result_ref != expected_source_subject
            or inventory.artifact_validation_result_refs
            != source_deterministic_validation.artifact_validation_result_refs
        ):
            raise ItemQualityPolicyError("candidate package inventory is stale")
        if semantic_workflow.workflow_result_sha256 != semantic_review_workflow_result_carried_sha256(
            semantic_workflow
        ):
            raise ItemQualityPolicyError("semantic workflow identity is stale")
        IsolatedSemanticReviewOrchestrator().validate_current(
            semantic_workflow,
            candidate_revision=candidate_revision,
            deterministic_validation=deterministic_validation,
            review_policy=review_policy,
            job_store=job_store,
            job_id=job_id,
            item_id=item_id,
        )

    def _finalize_package(
        self,
        *,
        semantic_workflow: SemanticReviewWorkflowResultV2,
        candidate_revision: AttachmentCandidateRevisionV2,
        deterministic_validation: RevisionDeterministicValidationV2,
        source_deterministic_validation: DeterministicItemValidationResultV2,
        audit: ContractAudit,
    ) -> _FinalPackageValues:
        inventory = source_deterministic_validation.candidate_inventory
        entries: tuple[CandidatePackageInventoryEntryV2, ...]
        inventory_ref: ObjectRef | None
        if inventory is None:
            if (
                candidate_revision.artifact_versions
                or candidate_revision.candidate_output_refs
                or deterministic_validation.artifact_validation_result_refs
            ):
                raise _PackageBlocked(ItemQualityFailureCodeV2.PACKAGE_INVENTORY_MISSING)
            entries = ()
            inventory_ref = None
        else:
            entries = inventory.entries
            inventory_ref = candidate_package_inventory_ref(inventory)
        contexts = _artifact_contexts(
            candidate_revision,
            source_deterministic_validation,
        )
        _validate_package_entries(
            entries,
            contexts,
        )
        candidate_ref = attachment_candidate_revision_ref(candidate_revision)
        validation_ref = revision_deterministic_validation_ref(deterministic_validation)
        source_ref = deterministic_item_validation_result_ref(source_deterministic_validation)
        manifest_audit_refs = (
            candidate_ref,
            validation_ref,
            source_ref,
            *((inventory_ref,) if inventory_ref else ()),
            *candidate_revision.artifact_version_refs,
            *candidate_revision.candidate_output_refs,
            *source_deterministic_validation.artifact_validation_result_refs,
            *(candidate_package_inventory_entry_ref(item) for item in entries),
        )
        manifest = FinalPackageManifestV2.create(
            candidate_revision_ref=candidate_ref,
            deterministic_validation_ref=validation_ref,
            source_deterministic_validation_ref=source_ref,
            candidate_inventory_ref=inventory_ref,
            artifact_version_refs=(candidate_revision.artifact_version_refs),
            output_refs=candidate_revision.candidate_output_refs,
            artifact_validation_result_refs=(source_deterministic_validation.artifact_validation_result_refs),
            entries=entries,
            accepted_artifact_refs=(candidate_revision.artifact_version_refs),
            audit=_safe_audit(audit, manifest_audit_refs),
        )
        provenance = _compile_provenance(
            manifest=manifest,
            semantic_workflow=semantic_workflow,
            candidate_revision=candidate_revision,
            deterministic_validation=deterministic_validation,
            entries=entries,
            contexts=contexts,
            audit=audit,
        )
        environment = _compile_environment(
            manifest=manifest,
            provenance=provenance,
            candidate_revision=candidate_revision,
            entries=entries,
            contexts=contexts,
            audit=audit,
        )
        return _FinalPackageValues(
            manifest=manifest,
            provenance=provenance,
            environment=environment,
        )

    @staticmethod
    def _result(
        *,
        task_contract_set: R4TaskContractSetV2,
        review_policy: SemanticReviewPolicyV2,
        semantic_workflow: SemanticReviewWorkflowResultV2,
        candidate_revision: AttachmentCandidateRevisionV2,
        deterministic_validation: RevisionDeterministicValidationV2,
        source_deterministic_validation: DeterministicItemValidationResultV2,
        outcome: ItemQualityOutcomeV2,
        failure_code: ItemQualityFailureCodeV2 | None,
        counts: tuple[int, int, int],
        final_values: _FinalPackageValues | None,
        audit: ContractAudit,
    ) -> ItemQualityCompilationResultV2:
        package_ref = final_package_manifest_ref(final_values.manifest) if final_values is not None else None
        provenance_ref = (
            provenance_manifest_v2_ref(final_values.provenance) if final_values is not None else None
        )
        environment_ref = (
            environment_spec_v2_ref(final_values.environment) if final_values is not None else None
        )
        candidate_inventory_ref = (
            candidate_package_inventory_ref(source_deterministic_validation.candidate_inventory)
            if source_deterministic_validation.candidate_inventory is not None
            else None
        )
        approvable = final_values is not None
        report_refs = (
            r4_task_contract_set_ref(task_contract_set),
            semantic_review_policy_ref(review_policy),
            semantic_review_workflow_result_ref(semantic_workflow),
            attachment_candidate_revision_ref(candidate_revision),
            revision_deterministic_validation_ref(deterministic_validation),
            deterministic_item_validation_result_ref(source_deterministic_validation),
            *((candidate_inventory_ref,) if candidate_inventory_ref else ()),
            *candidate_revision.artifact_version_refs,
            *candidate_revision.candidate_output_refs,
            *source_deterministic_validation.artifact_validation_result_refs,
            *semantic_workflow.round_result_refs,
            *semantic_workflow.stage_result_refs,
            *semantic_workflow.repair_plan_refs,
            *semantic_workflow.repair_result_refs,
            *semantic_workflow.current_finding_refs,
            *semantic_workflow.stale_finding_refs,
            *semantic_workflow.resolution_refs,
            *(candidate_revision.artifact_version_refs if approvable else ()),
            *((package_ref,) if package_ref else ()),
            *((provenance_ref,) if provenance_ref else ()),
            *((environment_ref,) if environment_ref else ()),
        )
        report = QualityReportV2.create(
            r4_task_contract_set_ref=r4_task_contract_set_ref(task_contract_set),
            review_policy_ref=semantic_review_policy_ref(review_policy),
            semantic_workflow_result_ref=(semantic_review_workflow_result_ref(semantic_workflow)),
            candidate_revision_ref=attachment_candidate_revision_ref(candidate_revision),
            deterministic_validation_ref=(revision_deterministic_validation_ref(deterministic_validation)),
            source_deterministic_validation_ref=(
                deterministic_item_validation_result_ref(source_deterministic_validation)
            ),
            candidate_inventory_ref=candidate_inventory_ref,
            artifact_version_refs=(candidate_revision.artifact_version_refs),
            output_refs=candidate_revision.candidate_output_refs,
            artifact_validation_result_refs=(source_deterministic_validation.artifact_validation_result_refs),
            semantic_round_result_refs=(semantic_workflow.round_result_refs),
            stage_result_refs=semantic_workflow.stage_result_refs,
            repair_plan_refs=semantic_workflow.repair_plan_refs,
            repair_result_refs=semantic_workflow.repair_result_refs,
            current_finding_refs=semantic_workflow.current_finding_refs,
            stale_finding_refs=semantic_workflow.stale_finding_refs,
            resolution_refs=semantic_workflow.resolution_refs,
            accepted_artifact_refs=(candidate_revision.artifact_version_refs if approvable else ()),
            final_package_manifest_ref=package_ref,
            provenance_manifest_ref=provenance_ref,
            environment_spec_ref=environment_ref,
            package_sha256=(final_values.manifest.package_sha256 if final_values is not None else None),
            input_state_only=True if approvable else None,
            outcome=outcome,
            failure_code=failure_code,
            open_p0_count=counts[0],
            open_p1_count=counts[1],
            unresolved_non_waivable_count=counts[2],
            approvable=approvable,
            audit=_safe_audit(audit, report_refs),
        )
        result_refs = (
            quality_report_v2_ref(report),
            *((package_ref,) if package_ref else ()),
            *((provenance_ref,) if provenance_ref else ()),
            *((environment_ref,) if environment_ref else ()),
            *report.accepted_artifact_refs,
        )
        return ItemQualityCompilationResultV2.create(
            quality_report=report,
            final_package_manifest=(final_values.manifest if final_values is not None else None),
            provenance_manifest=(final_values.provenance if final_values is not None else None),
            environment_spec=(final_values.environment if final_values is not None else None),
            audit=_safe_audit(audit, result_refs),
        )


def _finding_counts(
    workflow: SemanticReviewWorkflowResultV2,
    source: DeterministicItemValidationResultV2,
) -> tuple[int, int, int]:
    finding_by_ref: dict[
        ObjectRef,
        DeterministicValidationFindingV2 | SemanticReviewFindingV2,
    ] = {}
    for artifact_result in source.artifact_validation_results:
        for finding in artifact_result.findings:
            ref = deterministic_validation_finding_ref(finding)
            if ref in finding_by_ref:
                raise ItemQualityPolicyError("deterministic finding inventory contains duplicates")
            finding_by_ref[ref] = finding
    for semantic_finding in workflow.semantic_findings:
        ref = semantic_review_finding_ref(semantic_finding)
        if ref in finding_by_ref:
            raise ItemQualityPolicyError("quality finding inventory contains duplicate refs")
        finding_by_ref[ref] = semantic_finding
    current_values = []
    for ref in workflow.current_finding_refs:
        try:
            current_values.append(finding_by_ref[ref])
        except KeyError as exc:
            raise ItemQualityPolicyError("current finding ref lacks an authoritative value") from exc
    open_p0 = sum(item.severity is Severity.P0 for item in current_values)
    open_p1 = sum(item.severity is Severity.P1 for item in current_values)
    non_waivable = sum(item.non_waivable for item in current_values)
    return open_p0, open_p1, non_waivable


def _validate_passing_quality(
    *,
    semantic_workflow: SemanticReviewWorkflowResultV2,
    deterministic_validation: RevisionDeterministicValidationV2,
    source_deterministic_validation: DeterministicItemValidationResultV2,
    counts: tuple[int, int, int],
) -> None:
    if (
        semantic_workflow.current_finding_refs
        or any(counts)
        or len(semantic_workflow.round_results) != 3
        or not all(item.accepted for item in semantic_workflow.round_results)
        or deterministic_validation.outcome
        not in {
            RevisionDeterministicValidationOutcomeV2.PASSED,
            RevisionDeterministicValidationOutcomeV2.NOT_REQUIRED,
        }
        or source_deterministic_validation.outcome
        not in {
            DeterministicItemValidationOutcomeV2.PASSED,
            DeterministicItemValidationOutcomeV2.NOT_REQUIRED,
        }
    ):
        raise ItemQualityPolicyError("PASSED workflow does not satisfy item quality hard gates")


def _artifact_contexts(
    revision: AttachmentCandidateRevisionV2,
    source: DeterministicItemValidationResultV2,
) -> dict[str, _ArtifactContext]:
    artifacts = {item.artifact_id: item for item in revision.artifact_versions}
    if len(artifacts) != len(revision.artifact_versions):
        raise _PackageBlocked(ItemQualityFailureCodeV2.PACKAGE_INVENTORY_MISMATCH)
    contexts: dict[str, _ArtifactContext] = {}
    for validation in source.artifact_validation_results:
        artifact_id = validation.facade_validation_result.artifact_id
        artifact = artifacts.get(artifact_id)
        if artifact is None or artifact_id in contexts:
            raise _PackageBlocked(ItemQualityFailureCodeV2.PACKAGE_INVENTORY_MISMATCH)
        expected_subject = (
            artifact.base_artifact_build_result_ref
            if artifact.artifact_version == 1
            else candidate_artifact_version_ref(artifact)
        )
        if (
            validation.outcome is not ArtifactDeterministicValidationOutcomeV2.PASSED
            or validation.artifact_build_result_ref != expected_subject
            or validation.build_spec_ref != artifact.build_spec_ref
            or validation.output_ref != artifact.output_ref
        ):
            raise _PackageBlocked(ItemQualityFailureCodeV2.PACKAGE_INVENTORY_MISMATCH)
        validation_ref = ObjectRef(
            object_type="artifact-deterministic-validation-result",
            object_id=validation.artifact_validation_result_id,
            object_version="v2",
            object_sha256=validation.artifact_validation_result_sha256,
        )
        contexts[artifact_id] = _ArtifactContext(
            artifact=artifact,
            validation=validation,
            validation_ref=validation_ref,
        )
    if set(contexts) != set(artifacts):
        raise _PackageBlocked(ItemQualityFailureCodeV2.PACKAGE_INVENTORY_MISMATCH)
    return contexts


def _validate_package_entries(
    entries: tuple[CandidatePackageInventoryEntryV2, ...],
    contexts: dict[str, _ArtifactContext],
) -> None:
    validation_contexts = {item.validation_ref: item for item in contexts.values()}
    entries_by_validation: dict[ObjectRef, list[CandidatePackageInventoryEntryV2]] = {}
    observed_entry_refs: set[ObjectRef] = set()
    paths: set[tuple[str, str]] = set()
    casefold_paths: set[tuple[str, str]] = set()
    for entry in entries:
        entry_ref = candidate_package_inventory_entry_ref(entry)
        if entry_ref in observed_entry_refs:
            raise _PackageBlocked(ItemQualityFailureCodeV2.PACKAGE_MEMBER_INVALID)
        observed_entry_refs.add(entry_ref)
        context = validation_contexts.get(entry.artifact_validation_result_ref)
        if (
            context is None
            or entry.output_ref != context.artifact.output_ref
            or entry.build_spec_ref != context.artifact.build_spec_ref
        ):
            raise _PackageBlocked(ItemQualityFailureCodeV2.PACKAGE_INVENTORY_MISMATCH)
        entries_by_validation.setdefault(context.validation_ref, []).append(entry)
        member = entry.inventory_member
        if member.member_type in {"FILE", "NESTED_MEMBER"} and member.content_sha256 is None:
            raise _PackageBlocked(ItemQualityFailureCodeV2.PACKAGE_CONTENT_HASH_MISSING)
        container = entry.container_ref.object_id if entry.container_ref is not None else ""
        key = (container, member.normalized_path.rstrip("/"))
        casefold_key = (container, key[1].casefold())
        if key in paths or casefold_key in casefold_paths:
            raise _PackageBlocked(ItemQualityFailureCodeV2.PACKAGE_PATH_COLLISION)
        paths.add(key)
        casefold_paths.add(casefold_key)
    for context in contexts.values():
        allowed_containers = {context.artifact.output_ref}
        pending = [
            entry
            for entry in entries_by_validation.get(context.validation_ref, ())
            if entry.container_ref is not None
        ]
        while pending:
            unresolved: list[CandidatePackageInventoryEntryV2] = []
            for entry in pending:
                if entry.container_ref not in allowed_containers:
                    unresolved.append(entry)
                    continue
                member = entry.inventory_member
                if member.member_type == "NESTED_MEMBER" and member.content_sha256 is not None:
                    allowed_containers.add(_nested_container_ref(entry))
            if len(unresolved) == len(pending):
                raise _PackageBlocked(ItemQualityFailureCodeV2.PACKAGE_MEMBER_INVALID)
            pending = unresolved


def _nested_container_ref(
    entry: CandidatePackageInventoryEntryV2,
) -> ObjectRef:
    assert entry.container_ref is not None
    assert entry.inventory_member.content_sha256 is not None
    identity = hashlib.sha256(
        (f"{entry.container_ref.object_id}|{entry.inventory_member.normalized_path}").encode()
    ).hexdigest()
    return ObjectRef(
        object_type="attachment-output",
        object_id=f"attachment-output://container/{identity}",
        object_version="v2",
        object_sha256=entry.inventory_member.content_sha256,
    )


def _compile_provenance(
    *,
    manifest: FinalPackageManifestV2,
    semantic_workflow: SemanticReviewWorkflowResultV2,
    candidate_revision: AttachmentCandidateRevisionV2,
    deterministic_validation: RevisionDeterministicValidationV2,
    entries: tuple[CandidatePackageInventoryEntryV2, ...],
    contexts: dict[str, _ArtifactContext],
    audit: ContractAudit,
) -> ProvenanceManifestV2:
    package_ref = final_package_manifest_ref(manifest)
    candidate_ref = attachment_candidate_revision_ref(candidate_revision)
    deterministic_ref = revision_deterministic_validation_ref(deterministic_validation)
    workflow_ref = semantic_review_workflow_result_ref(semantic_workflow)
    validation_contexts = {item.validation_ref: item for item in contexts.values()}
    bindings: list[PackageMemberProvenanceV2] = []
    table = ProvenanceDecisionTable()
    for entry in entries:
        context = validation_contexts.get(entry.artifact_validation_result_ref)
        if context is None:
            raise _PackageBlocked(ItemQualityFailureCodeV2.PROVENANCE_INCOMPLETE)
        artifact_ref = candidate_artifact_version_ref(context.artifact)
        closure = _sorted_refs(
            (
                *entry.derivation_root_refs,
                artifact_ref,
                context.artifact.output_ref,
                context.artifact.build_spec_ref,
                context.validation_ref,
                candidate_ref,
                deterministic_ref,
                workflow_ref,
            )
        )
        decision_audit = _safe_audit(
            audit,
            (entry.inventory_member_ref, *closure),
        )
        decision = table.decide(
            ProvenanceDecisionInput(
                subject_ref=entry.inventory_member_ref,
                origin_class=OriginClass.SYSTEM_OR_HARNESS_CONTEXT,
                visibility=Visibility.CONTESTANT_VISIBLE,
                taint_labels=frozenset(),
                content_risk_labels=frozenset(),
                derived_from=closure,
                source_event_refs=(),
                rule_ids=("item-quality-package/r5-10-v1",),
                confidence=1.0,
                audit=decision_audit,
            ),
            requested_disposition=Disposition.ALLOW_INPUT_EVIDENCE,
        )
        decision_ref = provenance_decision_stable_ref(decision)
        binding_refs = (
            package_ref,
            candidate_package_inventory_entry_ref(entry),
            entry.inventory_member_ref,
            artifact_ref,
            context.artifact.output_ref,
            context.artifact.build_spec_ref,
            context.validation_ref,
            *closure,
            decision_ref,
        )
        try:
            binding = PackageMemberProvenanceV2.create(
                final_package_manifest_ref=package_ref,
                inventory_entry=entry,
                candidate_artifact_version_ref=artifact_ref,
                output_ref=context.artifact.output_ref,
                build_spec_ref=context.artifact.build_spec_ref,
                artifact_validation_result_ref=context.validation_ref,
                derivation_closure_refs=closure,
                provenance_decision=decision,
                audit=_safe_audit(audit, binding_refs),
            )
        except ValueError as exc:
            raise _PackageBlocked(ItemQualityFailureCodeV2.PROVENANCE_UNSAFE) from exc
        bindings.append(binding)
    binding_refs = tuple(
        package_member_provenance_ref(item)
        for item in sorted(
            bindings,
            key=lambda item: _ref_key(item.inventory_entry_ref),
        )
    )
    inventory_entry_refs = tuple(candidate_package_inventory_entry_ref(item) for item in manifest.entries)
    try:
        return ProvenanceManifestV2.create(
            final_package_manifest_ref=package_ref,
            candidate_revision_ref=candidate_ref,
            package_sha256=manifest.package_sha256,
            package_inventory_entry_refs=inventory_entry_refs,
            entries=tuple(bindings),
            audit=_safe_audit(
                audit,
                (
                    package_ref,
                    candidate_ref,
                    *inventory_entry_refs,
                    *binding_refs,
                ),
            ),
        )
    except ValueError as exc:
        raise _PackageBlocked(ItemQualityFailureCodeV2.PROVENANCE_INCOMPLETE) from exc


def _compile_environment(
    *,
    manifest: FinalPackageManifestV2,
    provenance: ProvenanceManifestV2,
    candidate_revision: AttachmentCandidateRevisionV2,
    entries: tuple[CandidatePackageInventoryEntryV2, ...],
    contexts: dict[str, _ArtifactContext],
    audit: ContractAudit,
) -> EnvironmentSpecV2:
    entries_by_validation: dict[
        ObjectRef,
        list[CandidatePackageInventoryEntryV2],
    ] = {}
    for entry in entries:
        entries_by_validation.setdefault(
            entry.artifact_validation_result_ref,
            [],
        ).append(entry)
    artifacts: list[EnvironmentArtifactV2] = []
    for artifact in candidate_revision.artifact_versions:
        context = contexts.get(artifact.artifact_id)
        if context is None:
            raise _PackageBlocked(ItemQualityFailureCodeV2.ENVIRONMENT_INCOMPLETE)
        owned_entries = tuple(
            sorted(
                entries_by_validation.get(context.validation_ref, []),
                key=lambda item: _ref_key(candidate_package_inventory_entry_ref(item)),
            )
        )
        roots = tuple(
            item
            for item in owned_entries
            if item.container_ref is None
            and item.inventory_member.normalized_path.rstrip("/") == artifact.logical_path.rstrip("/")
        )
        if len(roots) != 1:
            raise _PackageBlocked(ItemQualityFailureCodeV2.ENVIRONMENT_INCOMPLETE)
        artifacts.append(
            EnvironmentArtifactV2.create(
                artifact_id=artifact.artifact_id,
                attachment_dependency_id=(artifact.attachment_dependency_id),
                candidate_artifact_version_ref=(candidate_artifact_version_ref(artifact)),
                output_ref=artifact.output_ref,
                logical_path=artifact.logical_path,
                media_type=artifact.media_type,
                size_bytes=roots[0].inventory_member.size_bytes,
                artifact_validation_result_ref=context.validation_ref,
                package_inventory_entry_refs=tuple(
                    candidate_package_inventory_entry_ref(item) for item in owned_entries
                ),
            )
        )
    package_ref = final_package_manifest_ref(manifest)
    provenance_ref = provenance_manifest_v2_ref(provenance)
    candidate_ref = attachment_candidate_revision_ref(candidate_revision)
    artifact_refs = tuple(
        ObjectRef(
            object_type="environment-artifact",
            object_id=(f"environment-artifact://sha256/{item.artifact_sha256}"),
            object_version="v2",
            object_sha256=item.artifact_sha256,
        )
        for item in sorted(
            artifacts,
            key=lambda item: _ref_key(item.candidate_artifact_version_ref),
        )
    )
    try:
        return EnvironmentSpecV2.create(
            candidate_revision_ref=candidate_ref,
            final_package_manifest_ref=package_ref,
            provenance_manifest_ref=provenance_ref,
            candidate_artifact_version_refs=(candidate_revision.artifact_version_refs),
            artifacts=tuple(artifacts),
            package_sha256=manifest.package_sha256,
            audit=_safe_audit(
                audit,
                (
                    candidate_ref,
                    package_ref,
                    provenance_ref,
                    *candidate_revision.artifact_version_refs,
                    *artifact_refs,
                ),
            ),
        )
    except ValueError as exc:
        raise _PackageBlocked(ItemQualityFailureCodeV2.ENVIRONMENT_INCOMPLETE) from exc


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    governing = (
        *(item for item in audit.governing_versions if item.component != "item-quality"),
        VersionBinding(
            component="item-quality",
            version=ITEM_QUALITY_POLICY_VERSION,
        ),
    )
    return audit.model_copy(
        update={
            "governing_versions": governing,
            "input_refs": _sorted_refs(refs),
        }
    )


def _sorted_refs(
    refs: tuple[ObjectRef, ...],
) -> tuple[ObjectRef, ...]:
    unique = {_ref_key(ref): ref for ref in refs}
    return tuple(unique[key] for key in sorted(unique))


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _without_audit_actor_time(value: object) -> object:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if key == "audit" and isinstance(item, dict):
                normalized[key] = {
                    "schema_version": item.get("schema_version"),
                    "governing_versions": _without_audit_actor_time(item.get("governing_versions", [])),
                    "input_refs": _without_audit_actor_time(item.get("input_refs", [])),
                }
            else:
                normalized[key] = _without_audit_actor_time(item)
        return normalized
    if isinstance(value, list):
        return [_without_audit_actor_time(item) for item in value]
    return value
