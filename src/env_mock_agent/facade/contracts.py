from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, model_validator

Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
Identifier = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{2,255}$"),
]


def _validate_relative_path(value: str) -> str:
    if value.startswith("/") or any(part == ".." for part in value.split("/")):
        raise ValueError("path must be relative and cannot traverse parents")
    return value


RelativePath = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9._ -][A-Za-z0-9._/ -]{0,511}$"),
    AfterValidator(_validate_relative_path),
]


class FacadeModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class FacadeObjectRef(FacadeModel):
    object_type: Identifier
    object_id: Identifier
    object_version: str = Field(min_length=1, max_length=64)
    object_sha256: Sha256


class ReconstructionMode(StrEnum):
    TRACE_RICH = "TRACE_RICH"
    SKELETON_GUIDED = "SKELETON_GUIDED"
    PROMPT_ONLY = "PROMPT_ONLY"
    BLOCKED = "BLOCKED"


class AttachmentArtifactRequest(FacadeModel):
    artifact_id: Identifier
    build_spec_sha256: Sha256
    relative_path: RelativePath
    media_type: str = Field(min_length=3, max_length=255)
    mode: ReconstructionMode
    critical: bool
    content_contract_ref: FacadeObjectRef
    render_contract_ref: FacadeObjectRef
    evidence_refs: tuple[FacadeObjectRef, ...]
    provider_preference: tuple[Identifier, ...]
    runtime_preference: tuple[Identifier, ...]
    validator_ids: tuple[Identifier, ...] = Field(min_length=1)


class AttachmentReconstructionRequest(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-reconstruction-request/v1"] = (
        "env-mock-agent/attachment-reconstruction-request/v1"
    )
    request_id: Identifier
    task_id: Identifier
    producer_task_view_ref: FacadeObjectRef
    artifacts: tuple[AttachmentArtifactRequest, ...] = Field(min_length=1)
    profile: Literal["LH", "GENERIC", "CC"]
    profile_version: str = Field(min_length=1, max_length=128)
    idempotency_key: Identifier


class ArtifactResultStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"


class AttachmentArtifactResult(FacadeModel):
    artifact_id: Identifier
    build_spec_sha256: Sha256
    status: ArtifactResultStatus
    output_ref: FacadeObjectRef | None = None
    output_sha256: Sha256 | None = None
    validation_refs: tuple[FacadeObjectRef, ...] = ()
    finding_refs: tuple[FacadeObjectRef, ...] = ()
    provider: Identifier | None = None
    runtime: Identifier | None = None
    failure_code: Identifier | None = None
    failure_message: str | None = Field(default=None, max_length=2000)
    retryable: bool = False

    @model_validator(mode="after")
    def validate_outcome(self) -> AttachmentArtifactResult:
        if self.status is ArtifactResultStatus.SUCCEEDED:
            if self.output_ref is None or self.output_sha256 is None:
                raise ValueError("successful artifact result requires output reference and hash")
            if self.failure_code is not None:
                raise ValueError("successful artifact result cannot contain failure")
        elif self.failure_code is None:
            raise ValueError("failed or blocked artifact result requires failure_code")
        return self


class AttachmentReconstructionResult(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-reconstruction-result/v1"] = (
        "env-mock-agent/attachment-reconstruction-result/v1"
    )
    result_id: Identifier
    request_id: Identifier
    artifact_results: tuple[AttachmentArtifactResult, ...] = Field(min_length=1)
    accepted_artifact_ids: tuple[Identifier, ...]
    failed_artifact_ids: tuple[Identifier, ...]
    resumable_artifact_ids: tuple[Identifier, ...]
    package_ref: FacadeObjectRef | None = None
    package_sha256: Sha256 | None = None
    input_state_only: bool

    @model_validator(mode="after")
    def validate_sets(self) -> AttachmentReconstructionResult:
        result_ids = {item.artifact_id for item in self.artifact_results}
        classified = set(self.accepted_artifact_ids) | set(self.failed_artifact_ids)
        if result_ids != classified:
            raise ValueError("accepted and failed artifact IDs must exactly classify results")
        if not set(self.resumable_artifact_ids) <= set(self.failed_artifact_ids):
            raise ValueError("resumable artifact IDs must be failed artifacts")
        if self.package_ref is not None and self.package_sha256 is None:
            raise ValueError("package reference requires package hash")
        return self
