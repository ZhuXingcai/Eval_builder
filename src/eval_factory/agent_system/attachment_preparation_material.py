from __future__ import annotations

from typing import Literal

from env_mock_agent.facade import FacadeObjectRef
from eval_factory.agent_system.attachment_preparation import (
    AttachmentR5PreparationAuthority,
)
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectError,
    FactoryPrivateObjectStore,
)
from eval_factory.attachment_planning import (
    ApprovedSourceEvidenceResult,
    ApprovedSourceRetrievalExecutionResult,
    ApprovedSourceRetrievalRequest,
    ArtifactConsistencyDefinition,
    ArtifactEvidenceModeCompilationResult,
    ArtifactExecutionPlanningResult,
    ArtifactExecutionPreparation,
    ArtifactRoutingCompilationResult,
    ArtifactRoutingExecutionResult,
    ArtifactRoutingRequest,
    PromptOnlyDependencyDiscoveryProposal,
    PromptOnlyDependencyDiscoveryRequest,
    PromptOnlyDependencyDiscoveryResult,
    VerifiedPublicSourceChecks,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactEvidenceTargetV2,
    ArtifactRoutingPolicyV2,
    AttachmentPlanningContextV2,
    PromptOnlyDependencyPlanningContextV2,
    PublicSourceRetrievalPolicyV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.safety import EvidenceBundle
from eval_factory.contracts.task_v2 import (
    ProducerStorageAuthorizationV2,
    ProducerTaskViewV2,
    TaskDraftV2,
)
from eval_factory.provenance.timelines import FileVersionTimeline
from eval_factory.provenance.views import EvidenceViewResult


class FactoryAttachmentPreparationMaterialError(RuntimeError):
    pass


class _FactoryAttachmentPreparationMaterialV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-attachment-preparation-material/v1"] = (
        "eval-factory/private-attachment-preparation-material/v1"
    )
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
    verified_source_checks: tuple[
        VerifiedPublicSourceChecks,
        ...,
    ]
    retrieval_policy: PublicSourceRetrievalPolicyV2
    source_evidence_result: ApprovedSourceEvidenceResult
    routing_request: ArtifactRoutingRequest
    routing_execution_result: ArtifactRoutingExecutionResult
    routing_result: ArtifactRoutingCompilationResult
    routing_policy: ArtifactRoutingPolicyV2
    consistency_definitions: tuple[
        ArtifactConsistencyDefinition,
        ...,
    ]
    world_ledger_ref: FacadeObjectRef
    execution_preparation: ArtifactExecutionPreparation
    execution_result: ArtifactExecutionPlanningResult

    @classmethod
    def from_authority(
        cls,
        value: AttachmentR5PreparationAuthority,
    ) -> _FactoryAttachmentPreparationMaterialV1:
        return cls(
            task_draft=value.task_draft,
            producer_task_view=value.producer_task_view,
            storage_authorization=value.storage_authorization,
            producer_view_result=value.producer_view_result,
            evidence_bundle=value.evidence_bundle,
            attachment_planning_context=(value.attachment_planning_context),
            dependency_planning_context=(value.dependency_planning_context),
            dependency_request=value.dependency_request,
            dependency_proposal=value.dependency_proposal,
            existing_targets=value.existing_targets,
            dependency_result=value.dependency_result,
            timelines=value.timelines,
            evidence_mode_result=value.evidence_mode_result,
            retrieval_request=value.retrieval_request,
            retrieval_execution_result=(value.retrieval_execution_result),
            verified_source_checks=(value.verified_source_checks),
            retrieval_policy=value.retrieval_policy,
            source_evidence_result=(value.source_evidence_result),
            routing_request=value.routing_request,
            routing_execution_result=(value.routing_execution_result),
            routing_result=value.routing_result,
            routing_policy=value.routing_policy,
            consistency_definitions=(value.consistency_definitions),
            world_ledger_ref=value.world_ledger_ref,
            execution_preparation=(value.execution_preparation),
            execution_result=value.execution_result,
        )

    def to_authority(
        self,
    ) -> AttachmentR5PreparationAuthority:
        return AttachmentR5PreparationAuthority(
            task_draft=self.task_draft,
            producer_task_view=self.producer_task_view,
            storage_authorization=self.storage_authorization,
            producer_view_result=self.producer_view_result,
            evidence_bundle=self.evidence_bundle,
            attachment_planning_context=(self.attachment_planning_context),
            dependency_planning_context=(self.dependency_planning_context),
            dependency_request=self.dependency_request,
            dependency_proposal=self.dependency_proposal,
            existing_targets=self.existing_targets,
            dependency_result=self.dependency_result,
            timelines=self.timelines,
            evidence_mode_result=self.evidence_mode_result,
            retrieval_request=self.retrieval_request,
            retrieval_execution_result=(self.retrieval_execution_result),
            verified_source_checks=(self.verified_source_checks),
            retrieval_policy=self.retrieval_policy,
            source_evidence_result=self.source_evidence_result,
            routing_request=self.routing_request,
            routing_execution_result=(self.routing_execution_result),
            routing_result=self.routing_result,
            routing_policy=self.routing_policy,
            consistency_definitions=(self.consistency_definitions),
            world_ledger_ref=self.world_ledger_ref,
            execution_preparation=self.execution_preparation,
            execution_result=self.execution_result,
        )


class FactoryAttachmentPreparationMaterialStore:
    def __init__(
        self,
        private_store: FactoryPrivateObjectStore,
    ) -> None:
        self.private_store = private_store

    def put(
        self,
        authority: AttachmentR5PreparationAuthority,
    ) -> ObjectRef:
        return self.private_store.put_model(
            object_type="attachment-r5-preparation",
            value=(_FactoryAttachmentPreparationMaterialV1.from_authority(authority)),
        )

    def get(
        self,
        reference: ObjectRef,
    ) -> AttachmentR5PreparationAuthority:
        if (
            reference.object_type
            not in {
                "attachment-r5-preparation",
                "attachment-preparation-material",
            }
            or reference.object_version != "v2"
        ):
            raise FactoryAttachmentPreparationMaterialError(
                "attachment preparation material reference type is invalid"
            )
        try:
            value = self.private_store.get_model(
                reference,
                _FactoryAttachmentPreparationMaterialV1,
            )
        except FactoryPrivateObjectError as exc:
            raise FactoryAttachmentPreparationMaterialError(
                "attachment preparation material is unavailable"
            ) from exc
        return value.to_authority()


__all__ = [
    "FactoryAttachmentPreparationMaterialError",
    "FactoryAttachmentPreparationMaterialStore",
]
