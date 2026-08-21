from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    EvidenceRef,
    FailureRecord,
    Identifier,
    ObjectRef,
    RelativePath,
    Sha256,
)
from eval_factory.contracts.task import AttachmentCriticality


class ReconstructionMode(StrEnum):
    TRACE_RICH = "TRACE_RICH"
    SKELETON_GUIDED = "SKELETON_GUIDED"
    PROMPT_ONLY = "PROMPT_ONLY"
    BLOCKED = "BLOCKED"


class EvidenceCoverage(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class EvidenceStrength(StrEnum):
    DIRECT = "DIRECT"
    INFERRED = "INFERRED"
    MISSING = "MISSING"


class ArtifactEvidenceRow(ContractModel):
    schema_version: Literal["eval-factory/artifact-evidence-row/v1"] = "eval-factory/artifact-evidence-row/v1"
    artifact_id: Identifier
    logical_path: RelativePath
    media_type: str = Field(min_length=3, max_length=255)
    path_evidence: EvidenceStrength
    type_evidence: EvidenceStrength
    structure_coverage: EvidenceCoverage
    untainted_content_coverage: EvidenceCoverage
    pre_mutation_coverage: EvidenceCoverage
    provenance_confidence: float = Field(ge=0, le=1)
    truncation: Literal["NONE", "PRESENT", "UNKNOWN"]
    criticality: AttachmentCriticality
    blocking_uncertainties: tuple[Identifier, ...] = ()
    evidence: tuple[EvidenceRef, ...] = Field(min_length=1)
    selected_mode: ReconstructionMode

    @model_validator(mode="after")
    def validate_mode(self) -> ArtifactEvidenceRow:
        if (
            self.blocking_uncertainties
            and self.criticality is AttachmentCriticality.CRITICAL
            and self.selected_mode is not ReconstructionMode.BLOCKED
        ):
            raise ValueError("critical blocking uncertainty requires BLOCKED mode")
        if self.selected_mode is ReconstructionMode.TRACE_RICH and self.pre_mutation_coverage not in {
            EvidenceCoverage.COMPLETE,
            EvidenceCoverage.PARTIAL,
        }:
            raise ValueError("TRACE_RICH requires usable pre-mutation evidence")
        return self


class ArtifactEvidenceMatrix(ContractModel):
    schema_version: Literal["eval-factory/artifact-evidence-matrix/v1"] = (
        "eval-factory/artifact-evidence-matrix/v1"
    )
    artifact_evidence_matrix_id: Identifier
    producer_task_view_ref: ObjectRef
    rows: tuple[ArtifactEvidenceRow, ...] = Field(min_length=1)
    aggregate_mode: Literal[
        "TRACE_RICH",
        "SKELETON_GUIDED",
        "PROMPT_ONLY",
        "MIXED",
        "BLOCKED",
    ]
    audit: ContractAudit


class ArtifactBuildSpec(ContractModel):
    schema_version: Literal["eval-factory/artifact-build-spec/v1"] = "eval-factory/artifact-build-spec/v1"
    artifact_build_spec_id: Identifier
    artifact_evidence_row_ref: ObjectRef
    producer_task_view_ref: ObjectRef
    relative_path: RelativePath
    media_type: str = Field(min_length=3, max_length=255)
    mode: ReconstructionMode
    content_contract_ref: ObjectRef
    render_contract_ref: ObjectRef
    authorized_evidence_refs: tuple[EvidenceRef, ...]
    provider_preference: tuple[Identifier, ...]
    runtime_preference: tuple[Identifier, ...]
    validator_ids: tuple[Identifier, ...] = Field(min_length=1)
    retry_scope: Literal["ARTIFACT"]
    build_spec_sha256: Sha256
    audit: ContractAudit


class ArtifactBuildStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"


class ArtifactBuildResult(ContractModel):
    schema_version: Literal["eval-factory/artifact-build-result/v1"] = "eval-factory/artifact-build-result/v1"
    artifact_build_result_id: Identifier
    build_spec_ref: ObjectRef
    build_spec_sha256: Sha256
    status: ArtifactBuildStatus
    output_ref: ObjectRef | None = None
    output_sha256: Sha256 | None = None
    lineage_refs: tuple[ObjectRef, ...] = ()
    validation_result_refs: tuple[ObjectRef, ...] = ()
    finding_refs: tuple[ObjectRef, ...] = ()
    worker_version: str = Field(min_length=1, max_length=128)
    provider: Identifier | None = None
    runtime: Identifier | None = None
    failure: FailureRecord | None = None
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_outcome(self) -> ArtifactBuildResult:
        if self.status is ArtifactBuildStatus.SUCCEEDED:
            if self.output_ref is None or self.output_sha256 is None:
                raise ValueError("successful artifact build requires output reference and hash")
            if self.failure is not None:
                raise ValueError("successful artifact build cannot contain failure")
        elif self.failure is None:
            raise ValueError("non-success artifact build requires failure")
        return self


class AttachmentReconstructionResult(ContractModel):
    schema_version: Literal["eval-factory/attachment-reconstruction-result/v1"] = (
        "eval-factory/attachment-reconstruction-result/v1"
    )
    attachment_reconstruction_result_id: Identifier
    producer_task_view_ref: ObjectRef
    evidence_matrix_ref: ObjectRef
    artifact_result_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    accepted_artifact_refs: tuple[ObjectRef, ...]
    failed_artifact_ids: tuple[Identifier, ...]
    resumable_artifact_ids: tuple[Identifier, ...]
    environment_spec_ref: ObjectRef | None
    provenance_manifest_ref: ObjectRef | None
    quality_report_ref: ObjectRef | None
    package_sha256: Sha256 | None
    input_state_only: bool
    audit: ContractAudit
