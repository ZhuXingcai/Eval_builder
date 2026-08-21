from __future__ import annotations

import hashlib
from dataclasses import dataclass

from env_mock_agent.facade import (
    AttachmentExecutionFacade,
    AttachmentRoutingFacade,
    FacadeObjectRef,
)
from eval_factory.agent_system.attachment_preparation import (
    AttachmentR5PreparationAuthority,
    AttachmentR5PreparationValidator,
)
from eval_factory.agent_system.task_authoring_bridge import (
    FactoryTaskAuthoringBridgeResult,
)
from eval_factory.attachment_planning import (
    ApprovedSourceEvidenceCompiler,
    ApprovedSourceRetrievalRequestBuilder,
    ArtifactBuildContractDefinition,
    ArtifactBuildSpecCompiler,
    ArtifactConsistencyDefinition,
    ArtifactEvidenceModeCompiler,
    ArtifactExecutionPlanCompiler,
    ArtifactExecutionPreparationBuilder,
    ArtifactRoutingRequestBuilder,
    ArtifactRoutingRunner,
    AttachmentPlanningBridge,
    PromptOnlyDependencyCompiler,
    PromptOnlyDependencyOutcome,
    PromptOnlyDependencyPlanningProjector,
    PromptOnlyDependencyRequestBuilder,
    WorldLedgerSnapshotRunner,
)
from eval_factory.contracts.attachment_v2 import (
    ArtifactEvidenceTargetV2,
    ArtifactRoutingPolicyV2,
    PublicSourceRetrievalPolicyV2,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)


class FactoryAttachmentPreparationError(RuntimeError):
    pass


class FactoryAttachmentPreparationCapabilityError(FactoryAttachmentPreparationError):
    pass


@dataclass(frozen=True, slots=True)
class FactoryAttachmentPreparationConfig:
    retrieval_policy: PublicSourceRetrievalPolicyV2
    routing_policy: ArtifactRoutingPolicyV2
    world_ledger_ref: FacadeObjectRef
    dependency_model_profile: str = "internal-prompt-only-dependency-v1"
    dependency_prompt_version: str = "prompt-only-dependency/v1"


class FactoryAttachmentPreparationBuilder:
    def __init__(
        self,
        config: FactoryAttachmentPreparationConfig,
    ) -> None:
        self.config = config

    async def build(
        self,
        *,
        task_authoring: FactoryTaskAuthoringBridgeResult,
        routing_facade: AttachmentRoutingFacade,
        execution_facade: AttachmentExecutionFacade,
        audit: ContractAudit,
    ) -> AttachmentR5PreparationAuthority:
        task_draft = task_authoring.prompt_safety_result.task_draft
        producer_task_view = task_authoring.producer_task_view_result.producer_task_view
        storage_authorization = task_authoring.producer_task_view_result.storage_authorization
        if task_draft is None or producer_task_view is None or storage_authorization is None:
            raise FactoryAttachmentPreparationError("R4 authority is incomplete for attachment preparation")
        producer_view = task_authoring.producer_evidence_view
        evidence_bundle = task_authoring.producer_evidence_bundle
        context = AttachmentPlanningBridge().compile(
            producer_task_view=producer_task_view,
            storage_authorization=storage_authorization,
            evidence_bundle=evidence_bundle,
            audit=audit,
        )
        dependency_context = PromptOnlyDependencyPlanningProjector().compile(
            task_draft=task_draft,
            producer_task_view=producer_task_view,
            storage_authorization=storage_authorization,
            evidence_bundle=evidence_bundle,
            attachment_planning_context=context,
            audit=audit,
        )
        dependency_request = PromptOnlyDependencyRequestBuilder().build(
            attachment_planning_context=context,
            producer_task_view=producer_task_view,
            dependency_planning_context=(dependency_context),
            existing_targets=(),
            model_profile=(self.config.dependency_model_profile),
            prompt_version=(self.config.dependency_prompt_version),
            abstain_conditions=(
                "ambiguous input/output role",
                "media type unresolved",
            ),
            audit=audit,
        )
        dependency_result = PromptOnlyDependencyCompiler().compile(
            request=dependency_request,
            proposal=None,
            attachment_planning_context=context,
            producer_task_view=producer_task_view,
            dependency_planning_context=dependency_context,
            existing_targets=(),
            audit=audit,
        )
        if (
            dependency_result.outcome is not PromptOnlyDependencyOutcome.RESOLVED
            or dependency_result.discovery is None
        ):
            raise FactoryAttachmentPreparationCapabilityError(
                "attachment dependencies require semantic resolution"
            )
        discovery = dependency_result.discovery
        evidence_mode_result = ArtifactEvidenceModeCompiler().compile(
            attachment_planning_context=context,
            producer_task_view=producer_task_view,
            storage_authorization=storage_authorization,
            evidence_bundle=evidence_bundle,
            evidence_view_result=producer_view,
            timelines=(),
            targets=discovery.targets,
            audit=audit,
        )
        matrix = evidence_mode_result.artifact_evidence_matrix
        if matrix is None:
            raise FactoryAttachmentPreparationError("artifact evidence matrix did not compile")
        retrieval_request = ApprovedSourceRetrievalRequestBuilder().build(
            attachment_planning_context=context,
            producer_task_view=producer_task_view,
            storage_authorization=storage_authorization,
            producer_view_result=producer_view,
            producer_evidence_bundle=evidence_bundle,
            dependency_discovery=discovery,
            retrieval_policy=self.config.retrieval_policy,
            intent_definitions=(),
            audit=audit,
        )
        source_evidence_result = ApprovedSourceEvidenceCompiler().compile(
            request=retrieval_request,
            execution_result=None,
            verified_checks=(),
            attachment_planning_context=context,
            producer_task_view=producer_task_view,
            storage_authorization=storage_authorization,
            producer_view_result=producer_view,
            producer_evidence_bundle=evidence_bundle,
            dependency_discovery=discovery,
            retrieval_policy=(self.config.retrieval_policy),
            audit=audit,
        )
        definitions = tuple(self._build_contract_definition(target) for target in discovery.targets)
        routing_request = ArtifactRoutingRequestBuilder().build(
            attachment_planning_context=context,
            producer_task_view=producer_task_view,
            storage_authorization=storage_authorization,
            producer_view_result=producer_view,
            producer_evidence_bundle=evidence_bundle,
            artifact_evidence_matrix=matrix,
            artifact_targets=discovery.targets,
            routing_policy=self.config.routing_policy,
            contract_definitions=definitions,
            audit=audit,
        )
        routing_execution = await ArtifactRoutingRunner().run(
            routing_request,
            facade=routing_facade,
            audit=audit,
        )
        routing_result = ArtifactBuildSpecCompiler().compile(
            request=routing_request,
            execution_result=routing_execution,
            attachment_planning_context=context,
            producer_task_view=producer_task_view,
            storage_authorization=storage_authorization,
            producer_view_result=producer_view,
            producer_evidence_bundle=evidence_bundle,
            artifact_evidence_matrix=matrix,
            artifact_targets=discovery.targets,
            routing_policy=self.config.routing_policy,
            audit=audit,
        )
        consistency_definitions = tuple(
            ArtifactConsistencyDefinition(
                artifact_id=target.artifact_id,
                dependency_artifact_ids=(),
                locked_fact_ids=(),
            )
            for target in discovery.targets
        )
        execution_audit = _stage_audit(
            audit,
            component="artifact-execution",
            version="r5-06",
        )
        execution_preparation = ArtifactExecutionPreparationBuilder().build(
            routing_request=routing_request,
            routing_result=routing_result,
            definitions=consistency_definitions,
            world_ledger_ref=self.config.world_ledger_ref,
            audit=execution_audit,
        )
        snapshot = await WorldLedgerSnapshotRunner().run(
            execution_preparation,
            facade=execution_facade,
        )
        execution_result = ArtifactExecutionPlanCompiler().compile(
            routing_request=routing_request,
            routing_result=routing_result,
            preparation=execution_preparation,
            world_ledger_snapshot=snapshot,
            audit=execution_audit,
        )
        authority = AttachmentR5PreparationAuthority(
            task_draft=task_draft,
            producer_task_view=producer_task_view,
            storage_authorization=storage_authorization,
            producer_view_result=producer_view,
            evidence_bundle=evidence_bundle,
            attachment_planning_context=context,
            dependency_planning_context=dependency_context,
            dependency_request=dependency_request,
            dependency_proposal=None,
            existing_targets=(),
            dependency_result=dependency_result,
            timelines=(),
            evidence_mode_result=evidence_mode_result,
            retrieval_request=retrieval_request,
            retrieval_execution_result=None,
            verified_source_checks=(),
            retrieval_policy=self.config.retrieval_policy,
            source_evidence_result=source_evidence_result,
            routing_request=routing_request,
            routing_execution_result=routing_execution,
            routing_result=routing_result,
            routing_policy=self.config.routing_policy,
            consistency_definitions=consistency_definitions,
            world_ledger_ref=self.config.world_ledger_ref,
            execution_preparation=execution_preparation,
            execution_result=execution_result,
        )
        AttachmentR5PreparationValidator().validate_current(authority)
        return authority

    @staticmethod
    def _build_contract_definition(
        target: ArtifactEvidenceTargetV2,
    ) -> ArtifactBuildContractDefinition:
        if target.media_type != "text/plain":
            raise FactoryAttachmentPreparationCapabilityError("attachment media type is unsupported")
        suffix = hashlib.sha256(target.artifact_id.encode()).hexdigest()
        return ArtifactBuildContractDefinition(
            definition_id=(f"artifact-build-contract-definition://sha256/{suffix}"),
            attachment_dependency_id=(target.attachment_dependency_id),
            asset_type="txt",
            content_contract_ref=_stable_ref(
                "artifact-content-contract",
                f"{target.artifact_id}:content",
            ),
            render_contract_ref=_stable_ref(
                "artifact-render-contract",
                f"{target.artifact_id}:render",
            ),
            provider_payload_ref=_stable_ref(
                "attachment-provider-payload",
                f"{target.artifact_id}:payload",
            ),
            source_evidence_set_ref=None,
            required_provider_capability_ids=("attachment-provider/generate/txt/v1",),
            required_runtime_tools=("write",),
            runtime_role="attachment-writer",
            runtime_resume_required=True,
            validator_ids=(
                "secret-validator",
                "text-validator",
            ),
        )


def _stable_ref(
    object_type: str,
    seed: str,
) -> ObjectRef:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version="v2",
        object_sha256=digest,
    )


def _stage_audit(
    audit: ContractAudit,
    *,
    component: str,
    version: str,
) -> ContractAudit:
    return audit.model_copy(
        update={
            "governing_versions": (
                *tuple(binding for binding in audit.governing_versions if binding.component != component),
                VersionBinding(
                    component=component,
                    version=version,
                ),
            ),
        }
    )


__all__ = [
    "FactoryAttachmentPreparationBuilder",
    "FactoryAttachmentPreparationCapabilityError",
    "FactoryAttachmentPreparationConfig",
    "FactoryAttachmentPreparationError",
]
