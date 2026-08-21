from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import model_validator

from env_mock_agent.facade import AttachmentValidationFacade, FacadeObjectRef
from env_mock_agent.facade.execution_v2 import (
    attachment_execution_request_ref,
    attachment_execution_result_ref,
)
from env_mock_agent.facade.semantic_review_v2 import (
    AttachmentRepairOutcomeV2,
    AttachmentRepairRequestV2,
    AttachmentRepairSourceV2,
    AttachmentSemanticFindingResolutionV2,
    AttachmentSemanticReviewerRoleV2,
    AttachmentSemanticReviewFacade,
    AttachmentSemanticReviewFailureCodeV2,
    AttachmentSemanticReviewFindingV2,
    AttachmentSemanticReviewOutcomeV2,
    AttachmentSemanticReviewRequestV2,
    AttachmentSemanticReviewResultV2,
    AttachmentSemanticReviewRoundV2,
    attachment_repair_request_carried_sha256,
    attachment_repair_result_ref,
    attachment_semantic_finding_resolution_ref,
    attachment_semantic_review_finding_ref,
    attachment_semantic_review_request_carried_sha256,
    attachment_semantic_review_request_ref,
    attachment_semantic_review_result_ref,
    semantic_clean_context_attestation_ref,
    validate_attachment_semantic_review_result_identity,
)
from eval_factory.attachment_planning.validation import (
    DeterministicValidationCompiler,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactBuildResultOutcomeV2,
    AttachmentReconstructionResultV2,
    artifact_build_result_v2_ref,
    artifact_build_spec_v2_ref,
    attachment_reconstruction_result_v2_ref,
    validate_artifact_build_result_v2_identity,
    validate_attachment_reconstruction_result_v2_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    FailureClass,
    FailureRecord,
    Identifier,
    ObjectRef,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.orchestration import StageRunStatus
from eval_factory.contracts.orchestration_v2 import StageNameV2
from eval_factory.contracts.review_v2 import (
    AttachmentCandidateRevisionV2,
    CandidateArtifactVersionV2,
    CoverageSolvabilityReviewViewV2,
    DeterministicRepairTargetV2,
    LeakageExecutabilityReviewViewV2,
    RealismConsistencyReviewViewV2,
    RepairedArtifactBuildResultV2,
    RevisionDeterministicValidationOutcomeV2,
    RevisionDeterministicValidationV2,
    SemanticFindingResolutionV2,
    SemanticFindingScopeV2,
    SemanticReviewerRoleV2,
    SemanticReviewFindingV2,
    SemanticReviewPolicyV2,
    SemanticReviewRoundOutcomeV2,
    SemanticReviewRoundResultV2,
    SemanticReviewRoundV2,
    SemanticReviewWorkflowOutcomeV2,
    SemanticReviewWorkflowResultV2,
    TargetedRepairPlanV2,
    attachment_candidate_revision_ref,
    candidate_artifact_version_ref,
    create_leakage_executability_review_view,
    create_realism_consistency_review_view,
    repaired_artifact_build_result_ref,
    revision_deterministic_validation_ref,
    semantic_finding_resolution_is_current,
    semantic_finding_resolution_ref,
    semantic_review_context_view_ref,
    semantic_review_finding_is_current,
    semantic_review_finding_ref,
    semantic_review_policy_ref,
    semantic_review_round_result_carried_sha256,
    semantic_review_round_result_ref,
    semantic_review_workflow_result_carried_sha256,
    targeted_repair_plan_ref,
)
from eval_factory.contracts.task_v2 import (
    ProducerTaskViewV2,
    PromptLeakageReferenceSetV2,
)
from eval_factory.contracts.validation_v2 import (
    ArtifactDeterministicValidationOutcomeV2,
    DeterministicItemValidationResultV2,
    deterministic_item_validation_result_ref,
    validate_deterministic_item_validation_result_identity,
)
from eval_factory.orchestration import (
    JobStore,
    StageResultRecord,
    StageRunRecord,
    stage_result_record_ref,
    stage_run_record_ref,
)
from eval_factory.provenance.redaction import ConfiguredPiiRule


class SemanticReviewPolicyError(RuntimeError):
    pass


class _SemanticRepairBlocked(RuntimeError):
    def __init__(
        self,
        *,
        repair_plan_ref: ObjectRef,
        repair_result_refs: tuple[ObjectRef, ...],
    ) -> None:
        super().__init__("semantic repair did not produce a complete successor")
        self.repair_plan_ref = repair_plan_ref
        self.repair_result_refs = repair_result_refs


class SemanticReviewContextSources(ContractModelV2):
    schema_version: Literal["eval-factory/semantic-review-context-sources/r5-09"] = (
        "eval-factory/semantic-review-context-sources/r5-09"
    )
    query_public_view_ref: ObjectRef
    candidate_manifest_ref: ObjectRef
    public_rubric_requirement_refs: tuple[ObjectRef, ...]
    safe_evidence_bundle_refs: tuple[ObjectRef, ...]
    world_ledger_snapshot_ref: ObjectRef
    approved_source_evidence_refs: tuple[ObjectRef, ...]
    taint_lineage_projection_refs: tuple[ObjectRef, ...]
    evaluator_contract_ref: ObjectRef
    contestant_tool_policy_ref: ObjectRef
    leakage_reference_set_ref: ObjectRef
    executability_result_refs: tuple[ObjectRef, ...]

    @model_validator(mode="after")
    def validate_sources(self) -> SemanticReviewContextSources:
        for label, refs in (
            (
                "public rubric requirement refs",
                self.public_rubric_requirement_refs,
            ),
            ("safe evidence bundle refs", self.safe_evidence_bundle_refs),
            (
                "approved source evidence refs",
                self.approved_source_evidence_refs,
            ),
            (
                "taint lineage projection refs",
                self.taint_lineage_projection_refs,
            ),
            ("executability result refs", self.executability_result_refs),
        ):
            _require_sorted_unique_refs(label, refs)
        return self


class RevisionValidator(Protocol):
    async def validate(
        self,
        revision: AttachmentCandidateRevisionV2,
    ) -> RevisionDeterministicValidationV2: ...


class CompleteRevisionValidator:
    def __init__(
        self,
        *,
        producer_task_view: ProducerTaskViewV2,
        leakage_reference_set: PromptLeakageReferenceSetV2,
        configured_pii_rules: tuple[ConfiguredPiiRule, ...],
        facade: AttachmentValidationFacade,
        audit: ContractAudit,
    ) -> None:
        self._producer_task_view = producer_task_view
        self._leakage_reference_set = leakage_reference_set
        self._configured_pii_rules = configured_pii_rules
        self._facade = facade
        self._audit = audit
        self._compiler = DeterministicValidationCompiler()

    async def validate(
        self,
        revision: AttachmentCandidateRevisionV2,
    ) -> RevisionDeterministicValidationV2:
        source = await self._compiler.compile_revision(
            candidate_revision=revision,
            producer_task_view=self._producer_task_view,
            leakage_reference_set=self._leakage_reference_set,
            configured_pii_rules=self._configured_pii_rules,
            facade=self._facade,
            audit=self._audit,
        )
        revision_ref = attachment_candidate_revision_ref(revision)
        if source.attachment_reconstruction_result_ref != revision_ref:
            raise SemanticReviewPolicyError("revision validation source does not bind current candidate")
        return RevisionDeterministicValidationV2.create(
            candidate_revision_ref=revision_ref,
            source_deterministic_validation_result_ref=(deterministic_item_validation_result_ref(source)),
            artifact_validation_result_refs=(source.artifact_validation_result_refs),
            finding_refs=source.finding_refs,
            output_refs=revision.candidate_output_refs,
            outcome=RevisionDeterministicValidationOutcomeV2(source.outcome.value),
            audit=self._audit,
            repair_targets=_deterministic_repair_targets(source),
        )


@dataclass(frozen=True)
class _PendingSemanticResolution:
    facade_finding_ref: FacadeObjectRef
    factory_finding_ref: ObjectRef
    factory_finding: SemanticReviewFindingV2
    old_subject_refs: tuple[ObjectRef, ...]
    repair_plan_ref: ObjectRef
    facade_repair_result_refs: tuple[FacadeObjectRef, ...]


class CandidateRevisionCompiler:
    def compile_initial(
        self,
        *,
        reconstruction_result: AttachmentReconstructionResultV2,
        deterministic_validation: DeterministicItemValidationResultV2,
        audit: ContractAudit,
    ) -> tuple[
        AttachmentCandidateRevisionV2,
        RevisionDeterministicValidationV2,
    ]:
        validate_attachment_reconstruction_result_v2_identity(reconstruction_result)
        validate_deterministic_item_validation_result_identity(deterministic_validation)
        reconstruction_ref = _reconstruction_ref(reconstruction_result)
        if deterministic_validation.attachment_reconstruction_result_ref != reconstruction_ref:
            raise SemanticReviewPolicyError("deterministic validation belongs to another reconstruction")
        artifacts: list[CandidateArtifactVersionV2] = []
        for result in reconstruction_result.artifact_results:
            validate_artifact_build_result_v2_identity(result)
            if result.outcome is not ArtifactBuildResultOutcomeV2.SUCCEEDED:
                continue
            entry = result.route_entry
            receipt = result.execution_receipt
            if (
                entry.build_spec is None
                or receipt is None
                or receipt.facade_request is None
                or receipt.facade_result is None
                or receipt.facade_result.output_ref is None
            ):
                raise SemanticReviewPolicyError("successful artifact lacks exact review sources")
            build_spec = entry.build_spec
            artifacts.append(
                CandidateArtifactVersionV2.create_base(
                    artifact_id=entry.artifact_id,
                    attachment_dependency_id=entry.attachment_dependency_id,
                    artifact_build_result_ref=artifact_build_result_v2_ref(result),
                    build_spec_ref=artifact_build_spec_v2_ref(build_spec),
                    execution_request_ref=_object_ref_from_facade(
                        attachment_execution_request_ref(receipt.facade_request)
                    ),
                    execution_result_ref=_object_ref_from_facade(
                        attachment_execution_result_ref(receipt.facade_result)
                    ),
                    output_ref=_object_ref_from_facade(receipt.facade_result.output_ref),
                    logical_path=build_spec.build_spec.relative_path,
                    media_type=build_spec.build_spec.media_type,
                    declared_validator_ids=tuple(sorted(build_spec.build_spec.validator_ids)),
                    derivation_root_refs=_sorted_refs(
                        (
                            reconstruction_result.producer_task_view_ref,
                            artifact_build_result_v2_ref(result),
                            artifact_build_spec_v2_ref(build_spec),
                        )
                    ),
                )
            )
        revision = AttachmentCandidateRevisionV2.create_initial(
            base_reconstruction_result_ref=reconstruction_ref,
            artifact_versions=tuple(artifacts),
            audit=audit,
        )
        if revision.candidate_output_refs != reconstruction_result.candidate_output_refs:
            raise SemanticReviewPolicyError("candidate revision outputs do not match reconstruction")
        repair_targets: list[DeterministicRepairTargetV2] = []
        for artifact_result in deterministic_validation.artifact_validation_results:
            if artifact_result.outcome is not ArtifactDeterministicValidationOutcomeV2.REQUIRES_REPAIR:
                continue
            pairs = tuple(
                sorted(
                    zip(
                        artifact_result.finding_refs,
                        artifact_result.findings,
                        strict=True,
                    ),
                    key=lambda pair: _ref_key(pair[0]),
                )
            )
            repair_targets.append(
                DeterministicRepairTargetV2(
                    artifact_id=artifact_result.facade_validation_result.artifact_id,
                    finding_refs=tuple(pair[0] for pair in pairs),
                    finding_codes=tuple(pair[1].code.value for pair in pairs),
                )
            )
        wrapped = RevisionDeterministicValidationV2.create(
            candidate_revision_ref=attachment_candidate_revision_ref(revision),
            source_deterministic_validation_result_ref=(
                deterministic_item_validation_result_ref(deterministic_validation)
            ),
            artifact_validation_result_refs=(deterministic_validation.artifact_validation_result_refs),
            finding_refs=deterministic_validation.finding_refs,
            output_refs=revision.candidate_output_refs,
            outcome=RevisionDeterministicValidationOutcomeV2(deterministic_validation.outcome.value),
            audit=audit,
            repair_targets=tuple(repair_targets),
        )
        return revision, wrapped


ReviewView = (
    CoverageSolvabilityReviewViewV2 | RealismConsistencyReviewViewV2 | LeakageExecutabilityReviewViewV2
)


@dataclass(frozen=True)
class SemanticReviewRoundExecution:
    request: AttachmentSemanticReviewRequestV2
    facade_result: AttachmentSemanticReviewResultV2
    factory_findings: tuple[SemanticReviewFindingV2, ...]
    factory_resolutions: tuple[SemanticFindingResolutionV2, ...]
    round_result: SemanticReviewRoundResultV2


class SemanticReviewRoundExecutor:
    async def execute(
        self,
        *,
        running_stage: StageRunRecord,
        round_: SemanticReviewRoundV2,
        role: SemanticReviewerRoleV2,
        candidate_revision: AttachmentCandidateRevisionV2,
        deterministic_validation: RevisionDeterministicValidationV2,
        context_view: ReviewView,
        predecessor: SemanticReviewRoundResultV2 | None,
        prior_factory_findings: tuple[SemanticReviewFindingV2, ...],
        prior_view_finding_refs: tuple[ObjectRef, ...],
        prior_facade_finding_refs: tuple[FacadeObjectRef, ...],
        pending_resolutions: tuple[_PendingSemanticResolution, ...],
        prior_facade_resolution_refs: tuple[FacadeObjectRef, ...],
        prior_view_repair_plan_refs: tuple[ObjectRef, ...],
        prior_view_repair_result_refs: tuple[ObjectRef, ...],
        prior_facade_repair_result_refs: tuple[FacadeObjectRef, ...],
        review_policy: SemanticReviewPolicyV2,
        review_facade: AttachmentSemanticReviewFacade,
        audit: ContractAudit,
    ) -> SemanticReviewRoundExecution:
        if (
            running_stage.status is StageRunStatus.PENDING
            or running_stage.stage is not StageNameV2.ITEM_QUALITY
        ):
            raise SemanticReviewPolicyError(
                "semantic round executor requires a started ITEM_QUALITY StageRun"
            )
        view_ref = semantic_review_context_view_ref(context_view)
        request = _review_request(
            round_=round_,
            role=role,
            stage_run_ref=stage_run_record_ref(running_stage),
            candidate_revision=candidate_revision,
            deterministic_validation=deterministic_validation,
            context_view_ref=view_ref,
            prior_round_result=predecessor,
            prior_finding_refs=prior_facade_finding_refs,
            required_resolution_finding_refs=tuple(item.facade_finding_ref for item in pending_resolutions),
            prior_resolution_refs=prior_facade_resolution_refs,
            prior_repair_plan_refs=prior_view_repair_plan_refs,
            prior_repair_result_refs=(prior_facade_repair_result_refs),
            policy=review_policy,
        )
        facade_result = await review_facade.review(request)
        validate_attachment_semantic_review_result_identity(facade_result)
        facade_result_ref = _object_ref_from_facade(attachment_semantic_review_result_ref(facade_result))
        factory_findings = tuple(
            SemanticReviewFindingV2.create(
                facade_finding=item,
                facade_review_result_ref=facade_result_ref,
                audit=audit,
            )
            for item in facade_result.findings
        )
        finding_by_facade_ref = {
            _ref_key(item.facade_finding_ref): item
            for item in (
                *prior_factory_findings,
                *factory_findings,
            )
        }
        IsolatedSemanticReviewOrchestrator._validate_pending_resolutions(
            pending=pending_resolutions,
            facade_result=facade_result,
            candidate_revision=candidate_revision,
            deterministic_validation=deterministic_validation,
        )
        factory_resolutions: list[SemanticFindingResolutionV2] = []
        for resolution in facade_result.resolutions:
            prior_finding = finding_by_facade_ref.get(_facade_ref_key(resolution.prior_finding_ref))
            if prior_finding is None:
                raise SemanticReviewPolicyError("semantic resolution references unknown prior finding")
            successor = (
                finding_by_facade_ref.get(_facade_ref_key(resolution.successor_finding_ref))
                if resolution.successor_finding_ref is not None
                else None
            )
            factory_resolutions.append(
                SemanticFindingResolutionV2.create(
                    facade_resolution=resolution,
                    resolving_stage_run_ref=stage_run_record_ref(running_stage),
                    prior_finding_ref=semantic_review_finding_ref(prior_finding),
                    successor_finding_ref=(
                        semantic_review_finding_ref(successor) if successor is not None else None
                    ),
                    audit=audit,
                )
            )
        result_finding_refs = tuple(semantic_review_finding_ref(item) for item in factory_findings)
        result_resolution_refs = tuple(semantic_finding_resolution_ref(item) for item in factory_resolutions)
        round_result = _round_result(
            round_=round_,
            role=role,
            stage_run_ref=stage_run_record_ref(running_stage),
            context_view_ref=view_ref,
            candidate_revision=candidate_revision,
            deterministic_validation=deterministic_validation,
            request=request,
            result=facade_result,
            predecessor=predecessor,
            prior_finding_refs=prior_view_finding_refs,
            finding_refs=result_finding_refs,
            resolution_refs=result_resolution_refs,
            prior_repair_plan_refs=prior_view_repair_plan_refs,
            prior_repair_result_refs=prior_view_repair_result_refs,
            audit=audit,
        )
        return SemanticReviewRoundExecution(
            request=request,
            facade_result=facade_result,
            factory_findings=factory_findings,
            factory_resolutions=tuple(factory_resolutions),
            round_result=round_result,
        )


class SemanticReviewWorkflowCompiler:
    def compile(
        self,
        *,
        initial_revision: AttachmentCandidateRevisionV2,
        initial_validation: RevisionDeterministicValidationV2,
        review_policy: SemanticReviewPolicyV2,
        current_revision: AttachmentCandidateRevisionV2,
        validation_refs: tuple[ObjectRef, ...],
        repair_plan_refs: tuple[ObjectRef, ...],
        repair_result_refs: tuple[ObjectRef, ...],
        round_results: tuple[SemanticReviewRoundResultV2, ...],
        stage_result_refs: tuple[ObjectRef, ...],
        semantic_findings: tuple[SemanticReviewFindingV2, ...],
        semantic_resolutions: tuple[SemanticFindingResolutionV2, ...],
        current_finding_refs: tuple[ObjectRef, ...],
        stale_finding_refs: tuple[ObjectRef, ...],
        outcome: SemanticReviewWorkflowOutcomeV2,
        audit: ContractAudit,
    ) -> SemanticReviewWorkflowResultV2:
        return _workflow_result(
            initial_revision=initial_revision,
            initial_validation=initial_validation,
            review_policy=review_policy,
            current_revision=current_revision,
            validation_refs=validation_refs,
            repair_plan_refs=repair_plan_refs,
            repair_result_refs=repair_result_refs,
            round_results=round_results,
            stage_result_refs=stage_result_refs,
            semantic_findings=semantic_findings,
            semantic_resolutions=semantic_resolutions,
            current_finding_refs=current_finding_refs,
            stale_finding_refs=stale_finding_refs,
            outcome=outcome,
            audit=audit,
        )


class SemanticReviewContextCompiler:
    def compile(
        self,
        *,
        round_: SemanticReviewRoundV2,
        policy: SemanticReviewPolicyV2,
        candidate_revision: AttachmentCandidateRevisionV2,
        deterministic_validation: RevisionDeterministicValidationV2,
        sources: SemanticReviewContextSources,
        prior_round_result: SemanticReviewRoundResultV2 | None,
        prior_finding_refs: tuple[ObjectRef, ...],
        prior_resolution_refs: tuple[ObjectRef, ...],
        prior_repair_plan_refs: tuple[ObjectRef, ...],
        prior_repair_result_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> ReviewView:
        revision_ref = attachment_candidate_revision_ref(candidate_revision)
        validation_ref = revision_deterministic_validation_ref(deterministic_validation)
        artifact_refs = candidate_revision.artifact_version_refs
        output_refs = candidate_revision.candidate_output_refs
        if round_ is SemanticReviewRoundV2.COVERAGE_SOLVABILITY:
            if prior_round_result is not None:
                raise SemanticReviewPolicyError("coverage review cannot have predecessor")
            return CoverageSolvabilityReviewViewV2.create(
                candidate_revision_ref=revision_ref,
                deterministic_validation_result_ref=validation_ref,
                projection_policy_ref=policy.projection_for(round_),
                query_public_view_ref=sources.query_public_view_ref,
                candidate_manifest_ref=sources.candidate_manifest_ref,
                public_rubric_requirement_refs=(sources.public_rubric_requirement_refs),
                safe_evidence_bundle_refs=sources.safe_evidence_bundle_refs,
                deterministic_finding_refs=(deterministic_validation.finding_refs),
                audit=audit,
            )
        if prior_round_result is None:
            raise SemanticReviewPolicyError("later review requires predecessor")
        prior_ref = semantic_review_round_result_ref(prior_round_result)
        if round_ is SemanticReviewRoundV2.REALISM_CONSISTENCY:
            return create_realism_consistency_review_view(
                candidate_revision_ref=revision_ref,
                deterministic_validation_result_ref=validation_ref,
                projection_policy_ref=policy.projection_for(round_),
                current_artifact_version_refs=artifact_refs,
                current_output_refs=output_refs,
                world_ledger_snapshot_ref=sources.world_ledger_snapshot_ref,
                approved_source_evidence_refs=(sources.approved_source_evidence_refs),
                prior_round_result_ref=prior_ref,
                prior_finding_refs=prior_finding_refs,
                prior_resolution_refs=prior_resolution_refs,
                prior_repair_plan_refs=prior_repair_plan_refs,
                prior_repair_result_refs=prior_repair_result_refs,
                audit=audit,
            )
        return create_leakage_executability_review_view(
            candidate_revision_ref=revision_ref,
            deterministic_validation_result_ref=validation_ref,
            projection_policy_ref=policy.projection_for(round_),
            current_artifact_version_refs=artifact_refs,
            current_output_refs=output_refs,
            taint_lineage_projection_refs=(sources.taint_lineage_projection_refs),
            evaluator_contract_ref=sources.evaluator_contract_ref,
            contestant_tool_policy_ref=sources.contestant_tool_policy_ref,
            leakage_reference_set_ref=sources.leakage_reference_set_ref,
            executability_result_refs=sources.executability_result_refs,
            prior_round_result_ref=prior_ref,
            prior_finding_refs=prior_finding_refs,
            prior_resolution_refs=prior_resolution_refs,
            prior_repair_plan_refs=prior_repair_plan_refs,
            prior_repair_result_refs=prior_repair_result_refs,
            audit=audit,
        )


class IsolatedSemanticReviewOrchestrator:
    async def run(
        self,
        *,
        job_store: JobStore,
        job_id: Identifier,
        item_id: Identifier,
        candidate_revision: AttachmentCandidateRevisionV2,
        deterministic_validation: RevisionDeterministicValidationV2,
        review_policy: SemanticReviewPolicyV2,
        context_sources: SemanticReviewContextSources,
        review_facade: AttachmentSemanticReviewFacade,
        revision_validator: RevisionValidator,
        audit: ContractAudit,
    ) -> SemanticReviewWorkflowResultV2:
        initial_revision = candidate_revision
        initial_validation = deterministic_validation
        self._validate_current_subject(
            candidate_revision,
            deterministic_validation,
        )
        validation_refs: list[ObjectRef] = [revision_deterministic_validation_ref(deterministic_validation)]
        repair_plan_refs: list[ObjectRef] = []
        repair_result_refs: list[ObjectRef] = []
        stale_finding_refs: list[ObjectRef] = []
        admission = deterministic_validation.outcome
        if admission is RevisionDeterministicValidationOutcomeV2.REQUIRES_REPAIR:
            try:
                (
                    candidate_revision,
                    deterministic_validation,
                    repair_plan,
                    repaired_results,
                ) = await self._repair_deterministic_and_revalidate(
                    candidate_revision=candidate_revision,
                    deterministic_validation=deterministic_validation,
                    review_facade=review_facade,
                    revision_validator=revision_validator,
                    audit=audit,
                )
            except _SemanticRepairBlocked as exc:
                repair_plan_refs.append(exc.repair_plan_ref)
                repair_result_refs.extend(exc.repair_result_refs)
                admission = RevisionDeterministicValidationOutcomeV2.BLOCKED
            else:
                repair_plan_refs.append(targeted_repair_plan_ref(repair_plan))
                repair_result_refs.extend(
                    repaired_artifact_build_result_ref(item) for item in repaired_results
                )
                stale_finding_refs.extend(
                    ref for target in initial_validation.repair_targets for ref in target.finding_refs
                )
                validation_refs.append(revision_deterministic_validation_ref(deterministic_validation))
                admission = deterministic_validation.outcome
        if admission not in {
            RevisionDeterministicValidationOutcomeV2.PASSED,
            RevisionDeterministicValidationOutcomeV2.NOT_REQUIRED,
        }:
            return SemanticReviewWorkflowCompiler().compile(
                initial_revision=candidate_revision,
                initial_validation=initial_validation,
                review_policy=review_policy,
                current_revision=candidate_revision,
                validation_refs=tuple(validation_refs),
                repair_plan_refs=tuple(repair_plan_refs),
                repair_result_refs=tuple(repair_result_refs),
                round_results=(),
                stage_result_refs=(),
                semantic_findings=(),
                semantic_resolutions=(),
                current_finding_refs=deterministic_validation.finding_refs,
                stale_finding_refs=tuple(stale_finding_refs),
                outcome=_admission_outcome(admission),
                audit=audit,
            )

        compiler = SemanticReviewContextCompiler()
        round_results: list[SemanticReviewRoundResultV2] = []
        stage_result_refs: list[ObjectRef] = []
        all_findings: list[SemanticReviewFindingV2] = []
        all_resolutions: list[SemanticFindingResolutionV2] = []
        finding_by_facade_ref: dict[
            tuple[str, str, str, str],
            SemanticReviewFindingV2,
        ] = {}
        resolution_by_prior_facade_ref: dict[
            tuple[str, str, str, str],
            tuple[
                AttachmentSemanticFindingResolutionV2,
                SemanticFindingResolutionV2,
                SemanticReviewFindingV2,
            ],
        ] = {}
        prior_view_finding_refs: tuple[ObjectRef, ...] = ()
        prior_facade_finding_refs: tuple[FacadeObjectRef, ...] = ()
        prior_view_resolution_refs: tuple[ObjectRef, ...] = ()
        prior_facade_resolution_refs: tuple[FacadeObjectRef, ...] = ()
        prior_view_repair_plan_refs: tuple[ObjectRef, ...] = ()
        prior_view_repair_result_refs: tuple[ObjectRef, ...] = ()
        prior_facade_repair_result_refs: tuple[FacadeObjectRef, ...] = ()
        pending_resolutions: tuple[_PendingSemanticResolution, ...] = ()
        workflow_outcome = SemanticReviewWorkflowOutcomeV2.BLOCKED
        for round_ in review_policy.rounds_in_order:
            prior = round_results[-1] if round_results else None
            view = compiler.compile(
                round_=round_,
                policy=review_policy,
                candidate_revision=candidate_revision,
                deterministic_validation=deterministic_validation,
                sources=context_sources,
                prior_round_result=prior,
                prior_finding_refs=prior_view_finding_refs,
                prior_resolution_refs=prior_view_resolution_refs,
                prior_repair_plan_refs=prior_view_repair_plan_refs,
                prior_repair_result_refs=prior_view_repair_result_refs,
                audit=audit,
            )
            view_ref = semantic_review_context_view_ref(view)
            role = review_policy.roles_in_order[review_policy.rounds_in_order.index(round_)]
            stage = job_store.create_stage_run(
                stage_run_id=_stage_run_id(
                    job_id,
                    item_id,
                    round_,
                    attachment_candidate_revision_ref(candidate_revision),
                ),
                job_id=job_id,
                item_id=item_id,
                stage=StageNameV2.ITEM_QUALITY,
                attempt=1,
                principal_ref=_principal_ref(role),
                input_refs=_sorted_refs(
                    (
                        attachment_candidate_revision_ref(candidate_revision),
                        revision_deterministic_validation_ref(deterministic_validation),
                        view_ref,
                        *((semantic_review_round_result_ref(prior),) if prior is not None else ()),
                    )
                ),
                idempotency_key=(f"semantic-review-stage:{job_id}:{item_id}:{round_.value}"),
                retry_of_stage_run_id=None,
            )
            running = _ensure_stage_running(job_store, stage)
            execution = await SemanticReviewRoundExecutor().execute(
                running_stage=running,
                round_=round_,
                role=role,
                candidate_revision=candidate_revision,
                deterministic_validation=deterministic_validation,
                context_view=view,
                predecessor=prior,
                prior_factory_findings=tuple(all_findings),
                prior_view_finding_refs=prior_view_finding_refs,
                prior_facade_finding_refs=(prior_facade_finding_refs),
                pending_resolutions=pending_resolutions,
                prior_facade_resolution_refs=(prior_facade_resolution_refs),
                prior_view_repair_plan_refs=(prior_view_repair_plan_refs),
                prior_view_repair_result_refs=(prior_view_repair_result_refs),
                prior_facade_repair_result_refs=(prior_facade_repair_result_refs),
                review_policy=review_policy,
                review_facade=review_facade,
                audit=audit,
            )
            facade_result = execution.facade_result
            factory_findings = execution.factory_findings
            factory_resolutions = execution.factory_resolutions
            round_result = execution.round_result
            all_findings.extend(factory_findings)
            for finding in factory_findings:
                finding_by_facade_ref[_ref_key(finding.facade_finding_ref)] = finding
            result_finding_refs = tuple(semantic_review_finding_ref(item) for item in factory_findings)
            for resolution, factory_resolution in zip(
                facade_result.resolutions,
                factory_resolutions,
                strict=True,
            ):
                prior_finding = finding_by_facade_ref.get(_facade_ref_key(resolution.prior_finding_ref))
                if prior_finding is None:
                    raise SemanticReviewPolicyError("semantic resolution references unknown prior finding")
                resolution_by_prior_facade_ref[_facade_ref_key(resolution.prior_finding_ref)] = (
                    resolution,
                    factory_resolution,
                    prior_finding,
                )
            result_resolution_refs = tuple(
                semantic_finding_resolution_ref(item) for item in factory_resolutions
            )
            all_resolutions.extend(factory_resolutions)
            round_results.append(round_result)
            persisted = _complete_or_replay_stage(
                job_store=job_store,
                stage=running,
                round_result=round_result,
                facade_result=facade_result,
            )
            stage_result_refs.append(stage_result_record_ref(persisted))
            prior_view_finding_refs = _sorted_refs((*prior_view_finding_refs, *result_finding_refs))
            prior_facade_finding_refs = _sorted_facade_refs(
                (
                    *prior_facade_finding_refs,
                    *(attachment_semantic_review_finding_ref(item) for item in facade_result.findings),
                )
            )
            prior_view_resolution_refs = _sorted_refs((*prior_view_resolution_refs, *result_resolution_refs))
            prior_facade_resolution_refs = _sorted_facade_refs(
                (
                    *prior_facade_resolution_refs,
                    *(attachment_semantic_finding_resolution_ref(item) for item in facade_result.resolutions),
                )
            )
            pending_resolutions = ()
            workflow_outcome = _workflow_outcome(facade_result.outcome)
            if (
                facade_result.outcome is AttachmentSemanticReviewOutcomeV2.REQUIRES_REPAIR
                and round_ is not SemanticReviewRoundV2.LEAKAGE_EXECUTABILITY
            ):
                if not _semantic_findings_are_auto_repairable(facade_result.findings):
                    break
                try:
                    (
                        candidate_revision,
                        deterministic_validation,
                        repair_plan,
                        repaired_results,
                    ) = await self._repair_and_revalidate(
                        candidate_revision=candidate_revision,
                        deterministic_validation=deterministic_validation,
                        triggering_round_result=round_result,
                        facade_result=facade_result,
                        factory_findings=factory_findings,
                        review_facade=review_facade,
                        revision_validator=revision_validator,
                        audit=audit,
                    )
                except _SemanticRepairBlocked as exc:
                    repair_plan_refs.append(exc.repair_plan_ref)
                    repair_result_refs.extend(exc.repair_result_refs)
                    workflow_outcome = SemanticReviewWorkflowOutcomeV2.BLOCKED
                    break
                plan_ref = targeted_repair_plan_ref(repair_plan)
                repaired_refs = tuple(repaired_artifact_build_result_ref(item) for item in repaired_results)
                facade_repaired_refs = tuple(
                    attachment_repair_result_ref(item.facade_repair_result) for item in repaired_results
                )
                repair_plan_refs.append(plan_ref)
                repair_result_refs.extend(repaired_refs)
                prior_view_repair_plan_refs = _sorted_refs((*prior_view_repair_plan_refs, plan_ref))
                prior_view_repair_result_refs = _sorted_refs((*prior_view_repair_result_refs, *repaired_refs))
                prior_facade_repair_result_refs = _sorted_facade_refs(
                    (
                        *prior_facade_repair_result_refs,
                        *facade_repaired_refs,
                    )
                )
                stale_finding_refs.extend(result_finding_refs)
                validation_refs.append(revision_deterministic_validation_ref(deterministic_validation))
                pending_resolutions = self._pending_after_repair(
                    triggering_findings=factory_findings,
                    resolution_records=tuple(resolution_by_prior_facade_ref.values()),
                    candidate_revision=candidate_revision,
                    repair_plan_ref=plan_ref,
                    facade_repair_result_refs=facade_repaired_refs,
                )
                if deterministic_validation.outcome not in {
                    RevisionDeterministicValidationOutcomeV2.PASSED,
                    RevisionDeterministicValidationOutcomeV2.NOT_REQUIRED,
                }:
                    workflow_outcome = _admission_outcome(deterministic_validation.outcome)
                    break
                continue
            if facade_result.outcome is not AttachmentSemanticReviewOutcomeV2.ACCEPTED:
                break

        current_semantic_findings = tuple(
            semantic_review_finding_ref(item)
            for item in all_findings
            if semantic_review_finding_is_current(item, candidate_revision)
        )
        stale_semantic_findings = tuple(
            semantic_review_finding_ref(item)
            for item in all_findings
            if not semantic_review_finding_is_current(item, candidate_revision)
        )
        current_finding_refs = _sorted_refs(
            (
                *deterministic_validation.finding_refs,
                *current_semantic_findings,
            )
        )
        stale_finding_refs = list(_sorted_refs((*tuple(stale_finding_refs), *stale_semantic_findings)))
        if (
            len(round_results) == len(SemanticReviewRoundV2)
            and round_results[-1].accepted
            and not pending_resolutions
            and not any(
                _semantic_finding_is_blocking(item)
                for item in all_findings
                if semantic_review_finding_is_current(item, candidate_revision)
            )
            and all(
                semantic_finding_resolution_is_current(record[1], candidate_revision)
                for record in resolution_by_prior_facade_ref.values()
            )
        ):
            workflow_outcome = SemanticReviewWorkflowOutcomeV2.PASSED
        return SemanticReviewWorkflowCompiler().compile(
            initial_revision=initial_revision,
            initial_validation=initial_validation,
            review_policy=review_policy,
            current_revision=candidate_revision,
            validation_refs=tuple(validation_refs),
            repair_plan_refs=tuple(repair_plan_refs),
            repair_result_refs=tuple(repair_result_refs),
            round_results=tuple(round_results),
            stage_result_refs=tuple(stage_result_refs),
            semantic_findings=tuple(
                sorted(
                    all_findings,
                    key=lambda item: _ref_key(semantic_review_finding_ref(item)),
                )
            ),
            semantic_resolutions=tuple(
                sorted(
                    all_resolutions,
                    key=lambda item: _ref_key(semantic_finding_resolution_ref(item)),
                )
            ),
            current_finding_refs=current_finding_refs,
            stale_finding_refs=tuple(stale_finding_refs),
            outcome=workflow_outcome,
            audit=audit,
        )

    def validate_current(
        self,
        result: SemanticReviewWorkflowResultV2,
        *,
        candidate_revision: AttachmentCandidateRevisionV2,
        deterministic_validation: RevisionDeterministicValidationV2,
        review_policy: SemanticReviewPolicyV2,
        job_store: JobStore,
        job_id: Identifier,
        item_id: Identifier,
    ) -> None:
        try:
            validated = SemanticReviewWorkflowResultV2.model_validate(result.model_dump(mode="python"))
            if validated != result:
                raise SemanticReviewPolicyError("semantic workflow canonical payload changed")
            self._validate_current_subject(
                candidate_revision,
                deterministic_validation,
            )
            if (
                result.current_candidate_revision != candidate_revision
                or result.current_candidate_revision_ref
                != attachment_candidate_revision_ref(candidate_revision)
                or result.review_policy_ref != semantic_review_policy_ref(review_policy)
                or not result.deterministic_validation_result_refs
                or result.deterministic_validation_result_refs[-1]
                != revision_deterministic_validation_ref(deterministic_validation)
            ):
                raise SemanticReviewPolicyError("semantic workflow current candidate or validation is stale")
            if (
                tuple(item.round for item in result.round_results)
                != (review_policy.rounds_in_order[: len(result.round_results)])
                or tuple(item.reviewer_role for item in result.round_results)
                != (review_policy.roles_in_order[: len(result.round_results)])
            ):
                raise SemanticReviewPolicyError("semantic workflow round policy is stale")
            finding_by_ref = {semantic_review_finding_ref(item): item for item in result.semantic_findings}
            resolution_by_ref = {
                semantic_finding_resolution_ref(item): item for item in result.semantic_resolutions
            }
            if set(result.resolution_refs) != set(resolution_by_ref):
                raise SemanticReviewPolicyError("semantic workflow resolution inventory is incomplete")
            expected_current = _sorted_refs(
                (
                    *deterministic_validation.finding_refs,
                    *(
                        ref
                        for ref, finding in finding_by_ref.items()
                        if semantic_review_finding_is_current(
                            finding,
                            candidate_revision,
                        )
                    ),
                )
            )
            if result.current_finding_refs != expected_current:
                raise SemanticReviewPolicyError("semantic workflow current finding inventory is stale")
            expected_stale_semantic = {
                ref
                for ref, finding in finding_by_ref.items()
                if not semantic_review_finding_is_current(
                    finding,
                    candidate_revision,
                )
            }
            if not expected_stale_semantic <= set(result.stale_finding_refs):
                raise SemanticReviewPolicyError("semantic workflow omits stale semantic findings")
            current_resolutions = tuple(
                item
                for item in result.semantic_resolutions
                if semantic_finding_resolution_is_current(
                    item,
                    candidate_revision,
                )
            )
            if result.outcome is SemanticReviewWorkflowOutcomeV2.PASSED:
                resolved_prior_refs = {item.prior_finding_ref for item in current_resolutions}
                stale_blockers = {
                    ref
                    for ref, finding in finding_by_ref.items()
                    if ref in expected_stale_semantic and _semantic_finding_is_blocking(finding)
                }
                if stale_blockers - resolved_prior_refs or any(
                    _semantic_finding_is_blocking(finding)
                    for finding in finding_by_ref.values()
                    if semantic_review_finding_is_current(
                        finding,
                        candidate_revision,
                    )
                ):
                    raise SemanticReviewPolicyError(
                        "PASSED semantic workflow lacks current blocker resolution"
                    )
            runs = {
                item.stage_run_id: item
                for item in job_store.list_stage_runs(
                    job_id=job_id,
                    item_id=item_id,
                    stage=StageNameV2.ITEM_QUALITY,
                )
            }
            if len(runs) < len(result.round_results):
                raise SemanticReviewPolicyError("semantic workflow StageRun inventory is incomplete")
            for round_result, stage_result_ref in zip(
                result.round_results,
                result.stage_result_refs,
                strict=True,
            ):
                stage = runs.get(round_result.stage_run_ref.object_id)
                if stage is None or stage_run_record_ref(stage) != round_result.stage_run_ref:
                    raise SemanticReviewPolicyError("semantic workflow StageRun ref is stale")
                persisted = job_store.get_stage_result_for_run(stage.stage_run_id)
                if (
                    persisted is None
                    or stage_result_record_ref(persisted) != stage_result_ref
                    or persisted.output_refs != (semantic_review_round_result_ref(round_result),)
                ):
                    raise SemanticReviewPolicyError("semantic workflow StageResult ref is stale")
                incomplete = round_result.outcome in {
                    SemanticReviewRoundOutcomeV2.ABSTAINED,
                    SemanticReviewRoundOutcomeV2.BLOCKED,
                }
                if incomplete == (persisted.status is StageRunStatus.SUCCEEDED):
                    raise SemanticReviewPolicyError("semantic workflow StageResult status is inconsistent")
        except SemanticReviewPolicyError:
            raise
        except ValueError as exc:
            raise SemanticReviewPolicyError(f"current semantic workflow validation failed: {exc}") from exc

    @staticmethod
    def _validate_pending_resolutions(
        *,
        pending: tuple[_PendingSemanticResolution, ...],
        facade_result: AttachmentSemanticReviewResultV2,
        candidate_revision: AttachmentCandidateRevisionV2,
        deterministic_validation: RevisionDeterministicValidationV2,
    ) -> None:
        if facade_result.outcome in {
            AttachmentSemanticReviewOutcomeV2.ABSTAINED,
            AttachmentSemanticReviewOutcomeV2.BLOCKED,
        }:
            if facade_result.resolutions:
                raise SemanticReviewPolicyError("incomplete semantic review cannot resolve prior findings")
            return
        expected = {_facade_ref_key(item.facade_finding_ref): item for item in pending}
        observed = {_facade_ref_key(item.prior_finding_ref): item for item in facade_result.resolutions}
        if set(observed) != set(expected):
            raise SemanticReviewPolicyError("semantic review must resolve every repaired prior finding")
        validation_ref = _facade_ref(revision_deterministic_validation_ref(deterministic_validation))
        for key, pending_item in expected.items():
            resolution = observed[key]
            expected_current = tuple(
                _facade_ref(ref)
                for ref in _current_subject_refs_for_finding(
                    pending_item.factory_finding,
                    candidate_revision,
                )
            )
            if (
                resolution.old_subject_refs
                != tuple(_facade_ref(ref) for ref in pending_item.old_subject_refs)
                or resolution.current_subject_refs != expected_current
                or resolution.repair_plan_ref != _facade_ref(pending_item.repair_plan_ref)
                or resolution.repair_result_refs != pending_item.facade_repair_result_refs
                or resolution.deterministic_revalidation_ref != validation_ref
            ):
                raise SemanticReviewPolicyError("semantic resolution is stale for repaired finding")

    @staticmethod
    def _pending_after_repair(
        *,
        triggering_findings: tuple[SemanticReviewFindingV2, ...],
        resolution_records: tuple[
            tuple[
                AttachmentSemanticFindingResolutionV2,
                SemanticFindingResolutionV2,
                SemanticReviewFindingV2,
            ],
            ...,
        ],
        candidate_revision: AttachmentCandidateRevisionV2,
        repair_plan_ref: ObjectRef,
        facade_repair_result_refs: tuple[FacadeObjectRef, ...],
    ) -> tuple[_PendingSemanticResolution, ...]:
        pending_by_key: dict[
            tuple[str, str, str, str],
            _PendingSemanticResolution,
        ] = {}
        for finding in triggering_findings:
            if not _semantic_finding_is_blocking(finding):
                continue
            facade_ref = _facade_ref(finding.facade_finding_ref)
            pending_by_key[_facade_ref_key(facade_ref)] = _PendingSemanticResolution(
                facade_finding_ref=facade_ref,
                factory_finding_ref=semantic_review_finding_ref(finding),
                factory_finding=finding,
                old_subject_refs=finding.subject_refs,
                repair_plan_ref=repair_plan_ref,
                facade_repair_result_refs=facade_repair_result_refs,
            )
        for _facade_resolution, factory_resolution, prior_finding in resolution_records:
            if semantic_finding_resolution_is_current(
                factory_resolution,
                candidate_revision,
            ):
                continue
            facade_ref = _facade_ref(prior_finding.facade_finding_ref)
            pending_by_key[_facade_ref_key(facade_ref)] = _PendingSemanticResolution(
                facade_finding_ref=facade_ref,
                factory_finding_ref=semantic_review_finding_ref(prior_finding),
                factory_finding=prior_finding,
                old_subject_refs=factory_resolution.current_subject_refs,
                repair_plan_ref=repair_plan_ref,
                facade_repair_result_refs=facade_repair_result_refs,
            )
        return tuple(pending_by_key[key] for key in sorted(pending_by_key))

    @staticmethod
    def _validate_current_subject(
        revision: AttachmentCandidateRevisionV2,
        validation: RevisionDeterministicValidationV2,
    ) -> None:
        if (
            validation.candidate_revision_ref != attachment_candidate_revision_ref(revision)
            or validation.output_refs != revision.candidate_output_refs
        ):
            raise SemanticReviewPolicyError("deterministic validation is stale for candidate revision")

    async def _repair_deterministic_and_revalidate(
        self,
        *,
        candidate_revision: AttachmentCandidateRevisionV2,
        deterministic_validation: RevisionDeterministicValidationV2,
        review_facade: AttachmentSemanticReviewFacade,
        revision_validator: RevisionValidator,
        audit: ContractAudit,
    ) -> tuple[
        AttachmentCandidateRevisionV2,
        RevisionDeterministicValidationV2,
        TargetedRepairPlanV2,
        tuple[RepairedArtifactBuildResultV2, ...],
    ]:
        if not deterministic_validation.repair_targets:
            raise SemanticReviewPolicyError("deterministic repair outcome lacks exact targets")
        target_ids = tuple(item.artifact_id for item in deterministic_validation.repair_targets)
        finding_refs = tuple(
            ref for target in deterministic_validation.repair_targets for ref in target.finding_refs
        )
        untouched_refs = tuple(
            ref
            for artifact, ref in zip(
                candidate_revision.artifact_versions,
                candidate_revision.artifact_version_refs,
                strict=True,
            )
            if artifact.artifact_id not in target_ids
        )
        plan = TargetedRepairPlanV2.create(
            source_round=AttachmentRepairSourceV2.DETERMINISTIC,
            candidate_revision_ref=attachment_candidate_revision_ref(candidate_revision),
            deterministic_validation_result_ref=(
                revision_deterministic_validation_ref(deterministic_validation)
            ),
            triggering_round_result_ref=None,
            targeted_finding_refs=finding_refs,
            targeted_artifact_ids=target_ids,
            untouched_artifact_version_refs=untouched_refs,
            audit=audit,
        )
        artifact_by_id = {item.artifact_id: item for item in candidate_revision.artifact_versions}
        repair_inputs: list[
            tuple[
                DeterministicRepairTargetV2,
                CandidateArtifactVersionV2,
                AttachmentRepairRequestV2,
            ]
        ] = []
        for target in deterministic_validation.repair_targets:
            source = artifact_by_id.get(target.artifact_id)
            if source is None:
                raise SemanticReviewPolicyError("deterministic repair targets unknown artifact")
            request = _repair_request_values(
                plan=plan,
                source=source,
                candidate_revision=candidate_revision,
                finding_refs=target.finding_refs,
                finding_codes=target.finding_codes,
            )
            repair_inputs.append((target, source, request))
        results = tuple(
            await asyncio.gather(
                *(review_facade.repair(request) for _target, _source, request in repair_inputs)
            )
        )
        attempt_refs = tuple(
            _object_ref_from_facade(attachment_repair_result_ref(result)) for result in results
        )
        if any(result.outcome is not AttachmentRepairOutcomeV2.SUCCEEDED for result in results):
            raise _SemanticRepairBlocked(
                repair_plan_ref=targeted_repair_plan_ref(plan),
                repair_result_refs=attempt_refs,
            )
        repaired = tuple(
            RepairedArtifactBuildResultV2.create(
                source=source,
                repair_plan_ref=targeted_repair_plan_ref(plan),
                facade_request=request,
                facade_result=result,
                targeted_finding_refs=target.finding_refs,
                audit=audit,
            )
            for (target, source, request), result in zip(
                repair_inputs,
                results,
                strict=True,
            )
        )
        successor = AttachmentCandidateRevisionV2.create_successor(
            predecessor=candidate_revision,
            repair_plan=plan,
            repaired_results=repaired,
            audit=audit,
        )
        revalidated = await revision_validator.validate(successor)
        self._validate_current_subject(successor, revalidated)
        return successor, revalidated, plan, repaired

    async def _repair_and_revalidate(
        self,
        *,
        candidate_revision: AttachmentCandidateRevisionV2,
        deterministic_validation: RevisionDeterministicValidationV2,
        triggering_round_result: SemanticReviewRoundResultV2,
        facade_result: AttachmentSemanticReviewResultV2,
        factory_findings: tuple[SemanticReviewFindingV2, ...],
        review_facade: AttachmentSemanticReviewFacade,
        revision_validator: RevisionValidator,
        audit: ContractAudit,
    ) -> tuple[
        AttachmentCandidateRevisionV2,
        RevisionDeterministicValidationV2,
        TargetedRepairPlanV2,
        tuple[RepairedArtifactBuildResultV2, ...],
    ]:
        blocking_facade_findings = tuple(
            item for item in facade_result.findings if _facade_finding_is_blocking(item)
        )
        blocking_factory_findings = tuple(
            item for item in factory_findings if _semantic_finding_is_blocking(item)
        )
        if not _semantic_findings_are_auto_repairable(blocking_facade_findings):
            raise SemanticReviewPolicyError("semantic blockers are not eligible for artifact repair")
        target_ids = tuple(
            sorted(
                {artifact_id for finding in blocking_facade_findings for artifact_id in finding.artifact_ids}
            )
        )
        finding_refs = tuple(semantic_review_finding_ref(item) for item in blocking_factory_findings)
        untouched_refs = tuple(
            ref
            for artifact, ref in zip(
                candidate_revision.artifact_versions,
                candidate_revision.artifact_version_refs,
                strict=True,
            )
            if artifact.artifact_id not in target_ids
        )
        plan = TargetedRepairPlanV2.create(
            source_round=AttachmentRepairSourceV2(triggering_round_result.round.value),
            candidate_revision_ref=attachment_candidate_revision_ref(candidate_revision),
            deterministic_validation_result_ref=(
                revision_deterministic_validation_ref(deterministic_validation)
            ),
            triggering_round_result_ref=semantic_review_round_result_ref(triggering_round_result),
            targeted_finding_refs=finding_refs,
            targeted_artifact_ids=target_ids,
            untouched_artifact_version_refs=untouched_refs,
            audit=audit,
        )
        artifact_by_id = {item.artifact_id: item for item in candidate_revision.artifact_versions}
        repair_inputs: list[
            tuple[
                CandidateArtifactVersionV2,
                tuple[SemanticReviewFindingV2, ...],
                AttachmentRepairRequestV2,
            ]
        ] = []
        for artifact_id in target_ids:
            source = artifact_by_id.get(artifact_id)
            if source is None:
                raise SemanticReviewPolicyError("semantic repair targets unknown artifact")
            targeted_findings = tuple(
                item for item in blocking_facade_findings if artifact_id in item.artifact_ids
            )
            targeted_factory_findings = tuple(
                item for item in blocking_factory_findings if artifact_id in item.artifact_ids
            )
            request = _repair_request(
                plan=plan,
                source=source,
                candidate_revision=candidate_revision,
                findings=targeted_findings,
            )
            repair_inputs.append(
                (
                    source,
                    targeted_factory_findings,
                    request,
                )
            )
        results = tuple(
            await asyncio.gather(
                *(review_facade.repair(request) for _source, _findings, request in repair_inputs)
            )
        )
        attempt_refs = tuple(
            _object_ref_from_facade(attachment_repair_result_ref(result)) for result in results
        )
        if any(result.outcome is not AttachmentRepairOutcomeV2.SUCCEEDED for result in results):
            raise _SemanticRepairBlocked(
                repair_plan_ref=targeted_repair_plan_ref(plan),
                repair_result_refs=attempt_refs,
            )
        repaired = tuple(
            RepairedArtifactBuildResultV2.create(
                source=source,
                repair_plan_ref=targeted_repair_plan_ref(plan),
                facade_request=request,
                facade_result=result,
                targeted_finding_refs=tuple(
                    semantic_review_finding_ref(item) for item in targeted_factory_findings
                ),
                audit=audit,
            )
            for (
                source,
                targeted_factory_findings,
                request,
            ), result in zip(repair_inputs, results, strict=True)
        )
        successor = AttachmentCandidateRevisionV2.create_successor(
            predecessor=candidate_revision,
            repair_plan=plan,
            repaired_results=repaired,
            audit=audit,
        )
        revalidated = await revision_validator.validate(successor)
        self._validate_current_subject(successor, revalidated)
        return successor, revalidated, plan, repaired


def _review_request(
    *,
    round_: SemanticReviewRoundV2,
    role: SemanticReviewerRoleV2,
    stage_run_ref: ObjectRef,
    candidate_revision: AttachmentCandidateRevisionV2,
    deterministic_validation: RevisionDeterministicValidationV2,
    context_view_ref: ObjectRef,
    prior_round_result: SemanticReviewRoundResultV2 | None,
    prior_finding_refs: tuple[FacadeObjectRef, ...],
    required_resolution_finding_refs: tuple[FacadeObjectRef, ...],
    prior_resolution_refs: tuple[FacadeObjectRef, ...],
    prior_repair_plan_refs: tuple[ObjectRef, ...],
    prior_repair_result_refs: tuple[FacadeObjectRef, ...],
    policy: SemanticReviewPolicyV2,
) -> AttachmentSemanticReviewRequestV2:
    request = AttachmentSemanticReviewRequestV2(
        semantic_review_request_id="attachment-semantic-review-request://pending",
        round=AttachmentSemanticReviewRoundV2(round_.value),
        reviewer_role=AttachmentSemanticReviewerRoleV2(role.value),
        stage_run_ref=_facade_ref(stage_run_ref),
        candidate_revision_ref=_facade_ref(attachment_candidate_revision_ref(candidate_revision)),
        deterministic_validation_result_ref=_facade_ref(
            revision_deterministic_validation_ref(deterministic_validation)
        ),
        context_view_ref=_facade_ref(context_view_ref),
        current_artifact_version_refs=tuple(
            sorted(
                (_facade_ref(ref) for ref in candidate_revision.artifact_version_refs),
                key=_facade_ref_key,
            )
        ),
        current_output_refs=tuple(
            sorted(
                (_facade_ref(ref) for ref in candidate_revision.candidate_output_refs),
                key=_facade_ref_key,
            )
        ),
        prior_round_result_ref=(
            _facade_ref(semantic_review_round_result_ref(prior_round_result))
            if prior_round_result is not None
            else None
        ),
        prior_finding_refs=prior_finding_refs,
        required_resolution_finding_refs=required_resolution_finding_refs,
        prior_resolution_refs=prior_resolution_refs,
        prior_repair_plan_refs=tuple(_facade_ref(ref) for ref in prior_repair_plan_refs),
        prior_repair_result_refs=prior_repair_result_refs,
        model_profile_ref=_facade_ref(policy.model_profile_for(round_)),
        prompt_version=policy.prompt_for(round_),
        idempotency_key=(
            f"semantic-review-request:{round_.value}:{candidate_revision.candidate_revision_sha256}"
        ),
        semantic_review_request_sha256="0" * 64,
    )
    digest = attachment_semantic_review_request_carried_sha256(request)
    return request.model_copy(
        update={
            "semantic_review_request_id": (f"attachment-semantic-review-request://sha256/{digest}"),
            "semantic_review_request_sha256": digest,
        }
    )


def _repair_request(
    *,
    plan: TargetedRepairPlanV2,
    source: CandidateArtifactVersionV2,
    candidate_revision: AttachmentCandidateRevisionV2,
    findings: tuple[AttachmentSemanticReviewFindingV2, ...],
) -> AttachmentRepairRequestV2:
    return _repair_request_values(
        plan=plan,
        source=source,
        candidate_revision=candidate_revision,
        finding_refs=tuple(
            _object_ref_from_facade(attachment_semantic_review_finding_ref(item)) for item in findings
        ),
        finding_codes=tuple(item.code.value for item in findings),
    )


def _repair_request_values(
    *,
    plan: TargetedRepairPlanV2,
    source: CandidateArtifactVersionV2,
    candidate_revision: AttachmentCandidateRevisionV2,
    finding_refs: tuple[ObjectRef, ...],
    finding_codes: tuple[Identifier, ...],
) -> AttachmentRepairRequestV2:
    request = AttachmentRepairRequestV2(
        repair_request_id="attachment-repair-request://pending",
        repair_plan_ref=_facade_ref(targeted_repair_plan_ref(plan)),
        source_round=plan.source_round,
        candidate_revision_ref=_facade_ref(attachment_candidate_revision_ref(candidate_revision)),
        artifact_id=source.artifact_id,
        artifact_version_ref=_facade_ref(candidate_artifact_version_ref(source)),
        build_spec_ref=_facade_ref(source.build_spec_ref),
        execution_result_ref=_facade_ref(source.execution_result_ref),
        output_ref=_facade_ref(source.output_ref),
        output_sha256=source.output_sha256,
        targeted_finding_refs=tuple(_facade_ref(ref) for ref in finding_refs),
        targeted_finding_codes=finding_codes,
        attempt=1,
        prior_repair_result_ref=None,
        idempotency_key=(f"attachment-repair:{plan.plan_sha256}:{source.artifact_id}"),
        repair_request_sha256="0" * 64,
    )
    digest = attachment_repair_request_carried_sha256(request)
    return request.model_copy(
        update={
            "repair_request_id": (f"attachment-repair-request://sha256/{digest}"),
            "repair_request_sha256": digest,
        }
    )


def _round_result(
    *,
    round_: SemanticReviewRoundV2,
    role: SemanticReviewerRoleV2,
    stage_run_ref: ObjectRef,
    context_view_ref: ObjectRef,
    candidate_revision: AttachmentCandidateRevisionV2,
    deterministic_validation: RevisionDeterministicValidationV2,
    request: AttachmentSemanticReviewRequestV2,
    result: AttachmentSemanticReviewResultV2,
    predecessor: SemanticReviewRoundResultV2 | None,
    prior_finding_refs: tuple[ObjectRef, ...],
    finding_refs: tuple[ObjectRef, ...],
    resolution_refs: tuple[ObjectRef, ...],
    prior_repair_plan_refs: tuple[ObjectRef, ...],
    prior_repair_result_refs: tuple[ObjectRef, ...],
    audit: ContractAudit,
) -> SemanticReviewRoundResultV2:
    predecessor_ref = semantic_review_round_result_ref(predecessor) if predecessor is not None else None
    audit_refs = _sorted_refs(
        (
            stage_run_ref,
            context_view_ref,
            _object_ref_from_facade(semantic_clean_context_attestation_ref(result.clean_context_attestation)),
            _object_ref_from_facade(attachment_semantic_review_request_ref(request)),
            _object_ref_from_facade(attachment_semantic_review_result_ref(result)),
            attachment_candidate_revision_ref(candidate_revision),
            revision_deterministic_validation_ref(deterministic_validation),
            *prior_finding_refs,
            *finding_refs,
            *resolution_refs,
            *prior_repair_plan_refs,
            *prior_repair_result_refs,
            *((predecessor_ref,) if predecessor_ref is not None else ()),
        )
    )
    value = SemanticReviewRoundResultV2(
        semantic_review_round_result_id="semantic-review-round-result://pending",
        round=round_,
        reviewer_role=role,
        stage_run_ref=stage_run_ref,
        context_view_ref=context_view_ref,
        clean_context_attestation_ref=_object_ref_from_facade(
            semantic_clean_context_attestation_ref(result.clean_context_attestation)
        ),
        facade_review_request_ref=_object_ref_from_facade(attachment_semantic_review_request_ref(request)),
        facade_review_result_ref=_object_ref_from_facade(attachment_semantic_review_result_ref(result)),
        candidate_revision_ref=attachment_candidate_revision_ref(candidate_revision),
        deterministic_validation_result_ref=(revision_deterministic_validation_ref(deterministic_validation)),
        predecessor_round_result_ref=predecessor_ref,
        prior_finding_refs=_sorted_refs(prior_finding_refs),
        finding_refs=_sorted_refs(finding_refs),
        resolution_refs=_sorted_refs(resolution_refs),
        prior_repair_plan_refs=_sorted_refs(prior_repair_plan_refs),
        prior_repair_result_refs=_sorted_refs(prior_repair_result_refs),
        outcome=SemanticReviewRoundOutcomeV2(result.outcome.value),
        accepted=result.outcome is AttachmentSemanticReviewOutcomeV2.ACCEPTED,
        round_result_sha256="0" * 64,
        audit=audit.model_copy(update={"input_refs": audit_refs}),
    )
    digest = semantic_review_round_result_carried_sha256(value)
    return value.model_copy(
        update={
            "semantic_review_round_result_id": (f"semantic-review-round-result://sha256/{digest}"),
            "round_result_sha256": digest,
        }
    )


def _workflow_result(
    *,
    initial_revision: AttachmentCandidateRevisionV2,
    initial_validation: RevisionDeterministicValidationV2,
    review_policy: SemanticReviewPolicyV2,
    current_revision: AttachmentCandidateRevisionV2,
    validation_refs: tuple[ObjectRef, ...],
    repair_plan_refs: tuple[ObjectRef, ...],
    repair_result_refs: tuple[ObjectRef, ...],
    round_results: tuple[SemanticReviewRoundResultV2, ...],
    stage_result_refs: tuple[ObjectRef, ...],
    semantic_findings: tuple[SemanticReviewFindingV2, ...],
    semantic_resolutions: tuple[SemanticFindingResolutionV2, ...],
    current_finding_refs: tuple[ObjectRef, ...],
    stale_finding_refs: tuple[ObjectRef, ...],
    outcome: SemanticReviewWorkflowOutcomeV2,
    audit: ContractAudit,
) -> SemanticReviewWorkflowResultV2:
    review_policy_ref = semantic_review_policy_ref(review_policy)
    round_result_refs = tuple(semantic_review_round_result_ref(item) for item in round_results)
    resolution_refs = tuple(semantic_finding_resolution_ref(item) for item in semantic_resolutions)
    workflow_refs = _sorted_refs(
        (
            attachment_candidate_revision_ref(initial_revision),
            revision_deterministic_validation_ref(initial_validation),
            review_policy_ref,
            attachment_candidate_revision_ref(current_revision),
            *validation_refs,
            *repair_plan_refs,
            *repair_result_refs,
            *round_result_refs,
            *stage_result_refs,
            *current_finding_refs,
            *stale_finding_refs,
            *resolution_refs,
        )
    )
    value = SemanticReviewWorkflowResultV2(
        semantic_review_workflow_result_id=("semantic-review-workflow-result://pending"),
        initial_candidate_revision_ref=attachment_candidate_revision_ref(initial_revision),
        initial_deterministic_validation_result_ref=(
            revision_deterministic_validation_ref(initial_validation)
        ),
        review_policy_ref=review_policy_ref,
        current_candidate_revision=current_revision,
        current_candidate_revision_ref=attachment_candidate_revision_ref(current_revision),
        deterministic_validation_result_refs=validation_refs,
        repair_plan_refs=_sorted_refs(repair_plan_refs),
        repair_result_refs=_sorted_refs(repair_result_refs),
        round_results=round_results,
        round_result_refs=round_result_refs,
        stage_result_refs=stage_result_refs,
        semantic_findings=semantic_findings,
        semantic_resolutions=semantic_resolutions,
        current_finding_refs=_sorted_refs(current_finding_refs),
        stale_finding_refs=_sorted_refs(stale_finding_refs),
        resolution_refs=_sorted_refs(resolution_refs),
        outcome=outcome,
        accepted_artifact_refs=(),
        environment_spec_ref=None,
        provenance_manifest_ref=None,
        quality_report_ref=None,
        package_sha256=None,
        input_state_only=None,
        workflow_result_sha256="0" * 64,
        audit=audit.model_copy(update={"input_refs": workflow_refs}),
    )
    digest = semantic_review_workflow_result_carried_sha256(value)
    return value.model_copy(
        update={
            "semantic_review_workflow_result_id": (f"semantic-review-workflow-result://sha256/{digest}"),
            "workflow_result_sha256": digest,
        }
    )


def _admission_outcome(
    outcome: RevisionDeterministicValidationOutcomeV2,
) -> SemanticReviewWorkflowOutcomeV2:
    return {
        RevisionDeterministicValidationOutcomeV2.UPSTREAM_INCOMPLETE: (
            SemanticReviewWorkflowOutcomeV2.UPSTREAM_INCOMPLETE
        ),
        RevisionDeterministicValidationOutcomeV2.REJECTED: (SemanticReviewWorkflowOutcomeV2.REJECTED),
        RevisionDeterministicValidationOutcomeV2.BLOCKED: (SemanticReviewWorkflowOutcomeV2.BLOCKED),
        RevisionDeterministicValidationOutcomeV2.REQUIRES_REPAIR: (
            SemanticReviewWorkflowOutcomeV2.REQUIRES_REPAIR
        ),
    }[outcome]


def _workflow_outcome(
    outcome: AttachmentSemanticReviewOutcomeV2,
) -> SemanticReviewWorkflowOutcomeV2:
    return {
        AttachmentSemanticReviewOutcomeV2.ACCEPTED: (SemanticReviewWorkflowOutcomeV2.PASSED),
        AttachmentSemanticReviewOutcomeV2.REQUIRES_REPAIR: (SemanticReviewWorkflowOutcomeV2.REQUIRES_REPAIR),
        AttachmentSemanticReviewOutcomeV2.REJECTED: (SemanticReviewWorkflowOutcomeV2.REJECTED),
        AttachmentSemanticReviewOutcomeV2.ABSTAINED: (SemanticReviewWorkflowOutcomeV2.BLOCKED),
        AttachmentSemanticReviewOutcomeV2.BLOCKED: (SemanticReviewWorkflowOutcomeV2.BLOCKED),
    }[outcome]


def _deterministic_repair_targets(
    result: DeterministicItemValidationResultV2,
) -> tuple[DeterministicRepairTargetV2, ...]:
    targets: list[DeterministicRepairTargetV2] = []
    for artifact_result in result.artifact_validation_results:
        if artifact_result.outcome is not ArtifactDeterministicValidationOutcomeV2.REQUIRES_REPAIR:
            continue
        pairs = tuple(
            sorted(
                zip(
                    artifact_result.finding_refs,
                    artifact_result.findings,
                    strict=True,
                ),
                key=lambda pair: _ref_key(pair[0]),
            )
        )
        targets.append(
            DeterministicRepairTargetV2(
                artifact_id=artifact_result.facade_validation_result.artifact_id,
                finding_refs=tuple(pair[0] for pair in pairs),
                finding_codes=tuple(pair[1].code.value for pair in pairs),
            )
        )
    return tuple(sorted(targets, key=lambda item: item.artifact_id))


def _ensure_stage_running(
    job_store: JobStore,
    stage: StageRunRecord,
) -> StageRunRecord:
    if stage.status is StageRunStatus.PENDING:
        return job_store.transition_stage_run(
            stage.stage_run_id,
            StageRunStatus.RUNNING,
            expected_version=stage.row_version,
            idempotency_key=f"start:{stage.stage_run_id}",
        )
    return stage


def _complete_or_replay_stage(
    *,
    job_store: JobStore,
    stage: StageRunRecord,
    round_result: SemanticReviewRoundResultV2,
    facade_result: AttachmentSemanticReviewResultV2,
) -> StageResultRecord:
    output_refs = (semantic_review_round_result_ref(round_result),)
    status, failure = _semantic_stage_completion(facade_result)
    existing = job_store.get_stage_result_for_run(stage.stage_run_id)
    if existing is not None:
        if (
            existing.status is not status
            or existing.output_refs != output_refs
            or (existing.failure.code if existing.failure is not None else None)
            != (failure.code if failure is not None else None)
        ):
            raise SemanticReviewPolicyError("persisted semantic StageResult does not match replay")
        return existing
    if stage.status is not StageRunStatus.RUNNING:
        raise SemanticReviewPolicyError("semantic StageRun is terminal without immutable result")
    return job_store.complete_stage_run(
        stage_result_id=(
            f"stage-result://semantic-review/{_hash(round_result.semantic_review_round_result_id)}"
        ),
        stage_run_id=stage.stage_run_id,
        status=status,
        expected_version=stage.row_version,
        idempotency_key=f"complete:{stage.stage_run_id}",
        output_refs=output_refs,
        failure=failure,
    )


def _semantic_stage_completion(
    result: AttachmentSemanticReviewResultV2,
) -> tuple[StageRunStatus, FailureRecord | None]:
    if result.outcome not in {
        AttachmentSemanticReviewOutcomeV2.ABSTAINED,
        AttachmentSemanticReviewOutcomeV2.BLOCKED,
    }:
        return StageRunStatus.SUCCEEDED, None
    code = result.failure_code
    if code is None:
        raise SemanticReviewPolicyError("incomplete semantic review lacks failure code")
    if result.outcome is AttachmentSemanticReviewOutcomeV2.ABSTAINED:
        return (
            StageRunStatus.BLOCKED_CAPABILITY,
            FailureRecord(
                failure_class=FailureClass.INDETERMINATE,
                code=code.value,
                message="Semantic reviewer abstained without a complete conclusion.",
                retryable=False,
            ),
        )
    if code is AttachmentSemanticReviewFailureCodeV2.MODEL_UNAVAILABLE:
        return (
            StageRunStatus.BLOCKED_CAPABILITY,
            FailureRecord(
                failure_class=FailureClass.CAPABILITY,
                code=code.value,
                message="Semantic review capability was unavailable.",
                retryable=False,
            ),
        )
    if code is AttachmentSemanticReviewFailureCodeV2.BACKEND_FAILED:
        return (
            StageRunStatus.TERMINAL_FAILURE,
            FailureRecord(
                failure_class=FailureClass.INTERNAL,
                code=code.value,
                message="Semantic review backend did not complete.",
                retryable=False,
            ),
        )
    return (
        StageRunStatus.BLOCKED_POLICY,
        FailureRecord(
            failure_class=FailureClass.POLICY,
            code=code.value,
            message="Semantic review was blocked by context or policy.",
            retryable=False,
        ),
    )


def _facade_finding_is_blocking(
    finding: AttachmentSemanticReviewFindingV2,
) -> bool:
    return finding.non_waivable or finding.severity.value in {"P0", "P1"}


def _semantic_finding_is_blocking(
    finding: SemanticReviewFindingV2,
) -> bool:
    return finding.non_waivable or finding.severity.value in {"P0", "P1"}


def _semantic_findings_are_auto_repairable(
    findings: tuple[AttachmentSemanticReviewFindingV2, ...],
) -> bool:
    blockers = tuple(item for item in findings if _facade_finding_is_blocking(item))
    return bool(blockers) and all(
        not item.non_waivable and item.artifact_repair_allowed and bool(item.artifact_ids)
        for item in blockers
    )


def _current_subject_refs_for_finding(
    finding: SemanticReviewFindingV2,
    revision: AttachmentCandidateRevisionV2,
) -> tuple[ObjectRef, ...]:
    revision_ref = attachment_candidate_revision_ref(revision)
    if finding.scope is SemanticFindingScopeV2.ITEM:
        return (revision_ref,)
    artifact_ids = set(finding.artifact_ids)
    return _sorted_refs(
        (
            revision_ref,
            *(
                ref
                for artifact, ref in zip(
                    revision.artifact_versions,
                    revision.artifact_version_refs,
                    strict=True,
                )
                if artifact.artifact_id in artifact_ids
            ),
            *(
                artifact.output_ref
                for artifact in revision.artifact_versions
                if artifact.artifact_id in artifact_ids
            ),
        )
    )


def _principal_ref(role: SemanticReviewerRoleV2) -> ObjectRef:
    digest = _hash(role.value)
    return ObjectRef(
        object_type="stage-principal",
        object_id=f"stage-principal://semantic-review/{role.value.lower()}",
        object_version="r5-09-v1",
        object_sha256=digest,
    )


def _stage_run_id(
    job_id: str,
    item_id: str,
    round_: SemanticReviewRoundV2,
    revision_ref: ObjectRef,
) -> str:
    return f"stage-run://semantic-review/{_hash((job_id, item_id, round_.value, revision_ref.object_sha256))}"


def _facade_ref(ref: ObjectRef) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _object_ref_from_facade(ref: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _reconstruction_ref(
    value: AttachmentReconstructionResultV2,
) -> ObjectRef:
    return attachment_reconstruction_result_v2_ref(value)


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _facade_ref_key(
    ref: FacadeObjectRef,
) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _sorted_refs(refs: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    unique = {_ref_key(ref): ref for ref in refs}
    return tuple(unique[key] for key in sorted(unique))


def _sorted_facade_refs(
    refs: tuple[FacadeObjectRef, ...],
) -> tuple[FacadeObjectRef, ...]:
    unique = {_facade_ref_key(ref): ref for ref in refs}
    return tuple(unique[key] for key in sorted(unique))


def _require_sorted_unique_refs(
    label: str,
    refs: tuple[ObjectRef, ...],
) -> None:
    if refs != _sorted_refs(refs) or len(refs) != len(set(refs)):
        raise ValueError(f"{label} must be sorted and unique")


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
