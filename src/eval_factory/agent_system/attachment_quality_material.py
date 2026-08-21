from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from eval_factory.agent_system.attachment_finalization import (
    AttachmentR5FinalizationResult,
)
from eval_factory.agent_system.private_store import (
    FactoryPrivateObjectError,
    FactoryPrivateObjectStore,
)
from eval_factory.agent_system.solvability import (
    SolvabilityAgentResult,
)
from eval_factory.contracts.agent_system_v2 import (
    AttachmentQualityAssessmentV2,
    SolvabilityAssessmentV2,
)
from eval_factory.contracts.ai_gateway_v2 import (
    ModelRouteDecisionV2,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.quality_v2 import (
    ItemQualityCompilationResultV2,
)
from eval_factory.contracts.review_v2 import (
    AttachmentCandidateRevisionV2,
    RevisionDeterministicValidationV2,
    SemanticReviewWorkflowResultV2,
)
from eval_factory.contracts.validation_v2 import (
    DeterministicItemValidationResultV2,
)


class FactoryAttachmentQualityMaterialError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FactoryAttachmentQualityResult:
    finalization: AttachmentR5FinalizationResult
    solvability: SolvabilityAgentResult


class _FactoryAttachmentQualityMaterialV1(ContractModelV2):
    schema_version: Literal["eval-factory/private-attachment-quality-material/v1"] = (
        "eval-factory/private-attachment-quality-material/v1"
    )

    source_validation: DeterministicItemValidationResultV2
    candidate_revision: AttachmentCandidateRevisionV2
    deterministic_validation: RevisionDeterministicValidationV2
    semantic_workflow: SemanticReviewWorkflowResultV2
    item_quality: ItemQualityCompilationResultV2
    attachment_quality: AttachmentQualityAssessmentV2
    solvability_assessment: SolvabilityAssessmentV2
    solvability_route: ModelRouteDecisionV2
    solvability_invocation_result_ref: ObjectRef

    @classmethod
    def from_result(
        cls,
        result: FactoryAttachmentQualityResult,
    ) -> _FactoryAttachmentQualityMaterialV1:
        finalization = result.finalization
        return cls(
            source_validation=finalization.source_validation,
            candidate_revision=finalization.candidate_revision,
            deterministic_validation=(finalization.deterministic_validation),
            semantic_workflow=finalization.semantic_workflow,
            item_quality=finalization.item_quality,
            attachment_quality=finalization.attachment_quality,
            solvability_assessment=(result.solvability.assessment),
            solvability_route=result.solvability.route,
            solvability_invocation_result_ref=(result.solvability.invocation_result_ref),
        )

    def to_result(self) -> FactoryAttachmentQualityResult:
        return FactoryAttachmentQualityResult(
            finalization=AttachmentR5FinalizationResult(
                source_validation=self.source_validation,
                candidate_revision=self.candidate_revision,
                deterministic_validation=(self.deterministic_validation),
                semantic_workflow=self.semantic_workflow,
                item_quality=self.item_quality,
                attachment_quality=self.attachment_quality,
            ),
            solvability=SolvabilityAgentResult(
                assessment=self.solvability_assessment,
                route=self.solvability_route,
                invocation_result_ref=(self.solvability_invocation_result_ref),
            ),
        )


class FactoryAttachmentQualityMaterialStore:
    def __init__(
        self,
        private_store: FactoryPrivateObjectStore,
    ) -> None:
        self.private_store = private_store

    def put(
        self,
        result: FactoryAttachmentQualityResult,
    ) -> ObjectRef:
        return self.private_store.put_model(
            object_type="attachment-quality-material",
            value=_FactoryAttachmentQualityMaterialV1.from_result(result),
        )

    def get(
        self,
        reference: ObjectRef,
    ) -> FactoryAttachmentQualityResult:
        if reference.object_type != "attachment-quality-material" or reference.object_version != "v2":
            raise FactoryAttachmentQualityMaterialError(
                "attachment quality material reference type is invalid"
            )
        try:
            value = self.private_store.get_model(
                reference,
                _FactoryAttachmentQualityMaterialV1,
            )
        except FactoryPrivateObjectError as exc:
            raise FactoryAttachmentQualityMaterialError("attachment quality material is unavailable") from exc
        return value.to_result()


__all__ = [
    "FactoryAttachmentQualityMaterialError",
    "FactoryAttachmentQualityMaterialStore",
    "FactoryAttachmentQualityResult",
]
