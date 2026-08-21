from __future__ import annotations

from dataclasses import dataclass

from env_mock_agent.facade import FacadeObjectRef
from eval_factory.attachment_planning import (
    ApprovedSourceEvidenceCompiler,
    ApprovedSourceEvidenceOutcome,
    ApprovedSourceEvidencePolicyError,
    ApprovedSourceEvidenceResult,
    ApprovedSourceRetrievalExecutionResult,
    ApprovedSourceRetrievalRequest,
    ArtifactBuildSpecCompiler,
    ArtifactConsistencyDefinition,
    ArtifactEvidenceModeCompilationResult,
    ArtifactEvidenceModeCompiler,
    ArtifactEvidenceModeOutcome,
    ArtifactEvidenceModePolicyError,
    ArtifactExecutionPlanCompiler,
    ArtifactExecutionPlanningResult,
    ArtifactExecutionPolicyError,
    ArtifactExecutionPreparation,
    ArtifactExecutionPreparationBuilder,
    ArtifactRoutingCompilationResult,
    ArtifactRoutingExecutionResult,
    ArtifactRoutingPolicyError,
    ArtifactRoutingRequest,
    AttachmentPlanningBridge,
    AttachmentPlanningPolicyError,
    PromptOnlyDependencyCompiler,
    PromptOnlyDependencyDiscoveryProposal,
    PromptOnlyDependencyDiscoveryRequest,
    PromptOnlyDependencyDiscoveryResult,
    PromptOnlyDependencyOutcome,
    PromptOnlyDependencyPlanningProjector,
    PromptOnlyDependencyPolicyError,
    VerifiedPublicSourceChecks,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactEvidenceTargetV2,
    ArtifactRoutingPolicyV2,
    AttachmentPlanningContextV2,
    PromptOnlyDependencyPlanningContextV2,
    PublicSourceRetrievalPolicyV2,
    source_evidence_set_ref,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.safety import EvidenceBundle
from eval_factory.contracts.task_v2 import (
    ProducerStorageAuthorizationV2,
    ProducerTaskViewV2,
    TaskDraftV2,
)
from eval_factory.provenance.timelines import FileVersionTimeline
from eval_factory.provenance.views import EvidenceViewResult


class AttachmentR5PreparationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AttachmentR5PreparationAuthority:
    task_draft: TaskDraftV2
    producer_task_view: ProducerTaskViewV2
    storage_authorization: ProducerStorageAuthorizationV2
    producer_view_result: EvidenceViewResult
    evidence_bundle: EvidenceBundle
    attachment_planning_context: AttachmentPlanningContextV2
    dependency_planning_context: PromptOnlyDependencyPlanningContextV2
    dependency_request: PromptOnlyDependencyDiscoveryRequest
    dependency_proposal: PromptOnlyDependencyDiscoveryProposal | None
    existing_targets: tuple[ArtifactEvidenceTargetV2, ...]
    dependency_result: PromptOnlyDependencyDiscoveryResult
    timelines: tuple[FileVersionTimeline, ...]
    evidence_mode_result: ArtifactEvidenceModeCompilationResult
    retrieval_request: ApprovedSourceRetrievalRequest
    retrieval_execution_result: ApprovedSourceRetrievalExecutionResult | None
    verified_source_checks: tuple[VerifiedPublicSourceChecks, ...]
    retrieval_policy: PublicSourceRetrievalPolicyV2
    source_evidence_result: ApprovedSourceEvidenceResult
    routing_request: ArtifactRoutingRequest
    routing_execution_result: ArtifactRoutingExecutionResult
    routing_result: ArtifactRoutingCompilationResult
    routing_policy: ArtifactRoutingPolicyV2
    consistency_definitions: tuple[ArtifactConsistencyDefinition, ...]
    world_ledger_ref: FacadeObjectRef
    execution_preparation: ArtifactExecutionPreparation
    execution_result: ArtifactExecutionPlanningResult


class AttachmentR5PreparationValidator:
    def validate_current(
        self,
        authority: AttachmentR5PreparationAuthority,
    ) -> None:
        self._validate_planning_context(authority)
        self._validate_dependency_authority(authority)
        self._validate_evidence_modes(authority)
        self._validate_source_evidence(authority)
        self._validate_routing(authority)
        self._validate_execution_handoff(authority)

    @staticmethod
    def _validate_planning_context(
        authority: AttachmentR5PreparationAuthority,
    ) -> None:
        try:
            AttachmentPlanningBridge().validate_current(
                producer_task_view=authority.producer_task_view,
                storage_authorization=authority.storage_authorization,
                evidence_bundle=authority.evidence_bundle,
                attachment_planning_context=authority.attachment_planning_context,
            )
        except AttachmentPlanningPolicyError as exc:
            raise AttachmentR5PreparationError("R5-01 attachment planning authority is not current") from exc

    @staticmethod
    def _validate_dependency_authority(
        authority: AttachmentR5PreparationAuthority,
    ) -> None:
        try:
            PromptOnlyDependencyPlanningProjector().validate_current(
                task_draft=authority.task_draft,
                producer_task_view=authority.producer_task_view,
                storage_authorization=authority.storage_authorization,
                evidence_bundle=authority.evidence_bundle,
                attachment_planning_context=authority.attachment_planning_context,
                dependency_planning_context=authority.dependency_planning_context,
            )
            if (
                authority.dependency_result.outcome is not PromptOnlyDependencyOutcome.RESOLVED
                or authority.dependency_result.discovery is None
            ):
                raise AttachmentR5PreparationError("R5-03 dependency authority is not executable")
            PromptOnlyDependencyCompiler().validate_current(
                request=authority.dependency_request,
                proposal=authority.dependency_proposal,
                attachment_planning_context=authority.attachment_planning_context,
                producer_task_view=authority.producer_task_view,
                dependency_planning_context=authority.dependency_planning_context,
                existing_targets=authority.existing_targets,
                discovery=authority.dependency_result.discovery,
            )
        except PromptOnlyDependencyPolicyError as exc:
            raise AttachmentR5PreparationError("R5-03 dependency authority is not current") from exc

    @staticmethod
    def _validate_evidence_modes(
        authority: AttachmentR5PreparationAuthority,
    ) -> None:
        discovery = authority.dependency_result.discovery
        if discovery is None:
            raise AttachmentR5PreparationError("R5-03 dependency authority is missing")
        matrix = authority.evidence_mode_result.artifact_evidence_matrix
        if (
            authority.evidence_mode_result.outcome is not ArtifactEvidenceModeOutcome.COMPILED
            or matrix is None
        ):
            raise AttachmentR5PreparationError("R5-02 evidence authority is not executable")
        try:
            ArtifactEvidenceModeCompiler().validate_current(
                attachment_planning_context=authority.attachment_planning_context,
                producer_task_view=authority.producer_task_view,
                storage_authorization=authority.storage_authorization,
                evidence_bundle=authority.evidence_bundle,
                evidence_view_result=authority.producer_view_result,
                timelines=authority.timelines,
                targets=discovery.targets,
                artifact_evidence_matrix=matrix,
            )
        except ArtifactEvidenceModePolicyError as exc:
            raise AttachmentR5PreparationError("R5-02 artifact evidence authority is not current") from exc

    @staticmethod
    def _validate_source_evidence(
        authority: AttachmentR5PreparationAuthority,
    ) -> None:
        discovery = authority.dependency_result.discovery
        if discovery is None:
            raise AttachmentR5PreparationError("R5-03 dependency authority is missing")
        compiler = ApprovedSourceEvidenceCompiler()
        try:
            if authority.source_evidence_result.outcome is ApprovedSourceEvidenceOutcome.COMPILED:
                if (
                    authority.retrieval_execution_result is None
                    or authority.source_evidence_result.source_evidence_set is None
                ):
                    raise AttachmentR5PreparationError("R5-04 source evidence authority is incomplete")
                compiler.validate_current(
                    request=authority.retrieval_request,
                    execution_result=authority.retrieval_execution_result,
                    verified_checks=authority.verified_source_checks,
                    attachment_planning_context=authority.attachment_planning_context,
                    producer_task_view=authority.producer_task_view,
                    storage_authorization=authority.storage_authorization,
                    producer_view_result=authority.producer_view_result,
                    producer_evidence_bundle=authority.evidence_bundle,
                    dependency_discovery=discovery,
                    retrieval_policy=authority.retrieval_policy,
                    source_evidence_set=(authority.source_evidence_result.source_evidence_set),
                )
            elif authority.source_evidence_result.outcome is ApprovedSourceEvidenceOutcome.NOT_REQUIRED:
                rebuilt = compiler.compile(
                    request=authority.retrieval_request,
                    execution_result=authority.retrieval_execution_result,
                    verified_checks=authority.verified_source_checks,
                    attachment_planning_context=authority.attachment_planning_context,
                    producer_task_view=authority.producer_task_view,
                    storage_authorization=authority.storage_authorization,
                    producer_view_result=authority.producer_view_result,
                    producer_evidence_bundle=authority.evidence_bundle,
                    dependency_discovery=discovery,
                    retrieval_policy=authority.retrieval_policy,
                    audit=authority.source_evidence_result.audit,
                )
                if rebuilt != authority.source_evidence_result:
                    raise AttachmentR5PreparationError("R5-04 source evidence authority is not current")
            else:
                raise AttachmentR5PreparationError("R5-04 source evidence authority is not executable")
        except ApprovedSourceEvidencePolicyError as exc:
            raise AttachmentR5PreparationError("R5-04 source evidence authority is not current") from exc
        AttachmentR5PreparationValidator._validate_source_routing_join(
            authority,
        )

    @staticmethod
    def _validate_source_routing_join(
        authority: AttachmentR5PreparationAuthority,
    ) -> None:
        evidence_set = authority.source_evidence_result.source_evidence_set
        evidence_ref = source_evidence_set_ref(evidence_set) if evidence_set is not None else None
        covered_target_keys = (
            {_ref_key(ref) for ref in evidence_set.covered_target_refs} if evidence_set is not None else set()
        )
        for contract in authority.routing_request.build_contracts:
            expected_ref = (
                evidence_ref
                if _ref_key(contract.artifact_evidence_target_ref) in covered_target_keys
                else None
            )
            if contract.source_evidence_set_ref != expected_ref:
                raise AttachmentR5PreparationError(
                    "R5-04 source evidence does not match R5-05 build authority"
                )

    @staticmethod
    def _validate_routing(
        authority: AttachmentR5PreparationAuthority,
    ) -> None:
        matrix = authority.evidence_mode_result.artifact_evidence_matrix
        discovery = authority.dependency_result.discovery
        if matrix is None or discovery is None:
            raise AttachmentR5PreparationError("R5 routing inputs are incomplete")
        try:
            ArtifactBuildSpecCompiler().validate_current(
                request=authority.routing_request,
                execution_result=authority.routing_execution_result,
                compilation_result=authority.routing_result,
                attachment_planning_context=authority.attachment_planning_context,
                producer_task_view=authority.producer_task_view,
                storage_authorization=authority.storage_authorization,
                producer_view_result=authority.producer_view_result,
                producer_evidence_bundle=authority.evidence_bundle,
                artifact_evidence_matrix=matrix,
                artifact_targets=discovery.targets,
                routing_policy=authority.routing_policy,
            )
        except ArtifactRoutingPolicyError as exc:
            raise AttachmentR5PreparationError("R5-05 artifact routing authority is not current") from exc

    @staticmethod
    def _validate_execution_handoff(
        authority: AttachmentR5PreparationAuthority,
    ) -> None:
        try:
            rebuilt_preparation = ArtifactExecutionPreparationBuilder().build(
                routing_request=authority.routing_request,
                routing_result=authority.routing_result,
                definitions=authority.consistency_definitions,
                world_ledger_ref=authority.world_ledger_ref,
                audit=authority.execution_preparation.audit,
            )
            if rebuilt_preparation != authority.execution_preparation:
                raise AttachmentR5PreparationError("R5-06 execution preparation is not current")
            compiler = ArtifactExecutionPlanCompiler()
            rebuilt_result = compiler.compile(
                routing_request=authority.routing_request,
                routing_result=authority.routing_result,
                preparation=authority.execution_preparation,
                world_ledger_snapshot=authority.execution_result.world_ledger_snapshot,
                audit=authority.execution_result.audit,
            )
            if rebuilt_result != authority.execution_result:
                raise AttachmentR5PreparationError("R5-06 execution plan is not current")
            compiler.validate_current(authority.execution_result)
        except ArtifactExecutionPolicyError as exc:
            raise AttachmentR5PreparationError("R5-06 execution handoff is not current") from exc


def _ref_key(
    ref: ObjectRef,
) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


__all__ = [
    "AttachmentR5PreparationAuthority",
    "AttachmentR5PreparationError",
    "AttachmentR5PreparationValidator",
]
