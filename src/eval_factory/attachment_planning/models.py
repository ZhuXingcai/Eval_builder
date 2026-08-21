from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import field_validator, model_validator

from eval_factory.contracts.attachment_v2 import (
    ArtifactEvidenceMatrixV2,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.provenance.bundles import (
    EvidenceCompilationUncertainty,
)

ARTIFACT_EVIDENCE_MODE_POLICY_VERSION: Literal["artifact-evidence-mode/r5-02-v1"] = (
    "artifact-evidence-mode/r5-02-v1"
)


class AttachmentPlanningPolicyError(RuntimeError):
    pass


class ArtifactEvidenceModePolicyError(RuntimeError):
    pass


class ArtifactEvidenceModeOutcome(StrEnum):
    COMPILED = "COMPILED"
    NOT_REQUIRED = "NOT_REQUIRED"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"


class ArtifactEvidenceModeReason(StrEnum):
    MISSING_ARTIFACT_TARGET = "MISSING_ARTIFACT_TARGET"


class ArtifactEvidenceModeCompilationResult(ContractModel):
    schema_version: Literal["eval-factory/artifact-evidence-mode-compilation-result/r5-02"] = (
        "eval-factory/artifact-evidence-mode-compilation-result/r5-02"
    )
    result_id: Identifier
    attachment_planning_context_ref: ObjectRef
    outcome: ArtifactEvidenceModeOutcome
    artifact_evidence_matrix: ArtifactEvidenceMatrixV2 | None
    missing_dependency_ids: tuple[Identifier, ...] = ()
    reasons: frozenset[ArtifactEvidenceModeReason] = frozenset()
    uncertainties: tuple[EvidenceCompilationUncertainty, ...] = ()
    policy_version: Literal["artifact-evidence-mode/r5-02-v1"] = ARTIFACT_EVIDENCE_MODE_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> ArtifactEvidenceModeOutcome:
        if isinstance(value, ArtifactEvidenceModeOutcome):
            return value
        if isinstance(value, str):
            return ArtifactEvidenceModeOutcome(value)
        raise TypeError("outcome must be an ArtifactEvidenceModeOutcome")

    @field_validator("reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[ArtifactEvidenceModeReason]:
        if not isinstance(value, (tuple, list, set, frozenset)):
            raise TypeError("reasons must be a collection")
        return frozenset(ArtifactEvidenceModeReason(item) for item in value)

    @field_validator("uncertainties", mode="before")
    @classmethod
    def parse_uncertainties(
        cls,
        value: object,
    ) -> tuple[EvidenceCompilationUncertainty, ...]:
        if not isinstance(value, (tuple, list, set, frozenset)):
            raise TypeError("uncertainties must be a collection")
        return tuple(EvidenceCompilationUncertainty(item) for item in value)

    @model_validator(mode="after")
    def validate_result(self) -> ArtifactEvidenceModeCompilationResult:
        _require_context_ref(self.attachment_planning_context_ref)
        if self.missing_dependency_ids != tuple(sorted(set(self.missing_dependency_ids))):
            raise ValueError("missing_dependency_ids must be unique and sorted")
        if self.uncertainties != tuple(
            sorted(
                set(self.uncertainties),
                key=lambda item: item.value,
            )
        ):
            raise ValueError("uncertainties must be unique and sorted")
        if self.outcome is ArtifactEvidenceModeOutcome.COMPILED:
            if self.artifact_evidence_matrix is None:
                raise ValueError("COMPILED requires an evidence matrix")
            if (
                self.artifact_evidence_matrix.attachment_planning_context_ref
                != self.attachment_planning_context_ref
            ):
                raise ValueError("COMPILED matrix must bind the result planning context")
            if self.artifact_evidence_matrix.policy_version != self.policy_version:
                raise ValueError("COMPILED matrix must use the result policy version")
            if self.missing_dependency_ids or self.reasons:
                raise ValueError("COMPILED cannot contain missing targets or reasons")
        elif self.outcome is ArtifactEvidenceModeOutcome.NOT_REQUIRED:
            if (
                self.artifact_evidence_matrix is not None
                or self.missing_dependency_ids
                or self.reasons
                or self.uncertainties
            ):
                raise ValueError("NOT_REQUIRED cannot contain matrix, gaps, or reasons")
        else:
            if self.artifact_evidence_matrix is not None:
                raise ValueError("BLOCKED_CAPABILITY cannot contain a partial matrix")
            if not self.missing_dependency_ids:
                raise ValueError("BLOCKED_CAPABILITY requires missing dependency IDs")
            if self.reasons != frozenset({ArtifactEvidenceModeReason.MISSING_ARTIFACT_TARGET}):
                raise ValueError("BLOCKED_CAPABILITY requires the missing-target reason")
            if self.uncertainties:
                raise ValueError("target capability blocking cannot claim row uncertainties")
        return self


def _require_context_ref(ref: ObjectRef) -> None:
    if ref.object_type != "attachment-planning-context" or ref.object_version != "v2":
        raise ValueError("attachment_planning_context_ref must reference context v2")
