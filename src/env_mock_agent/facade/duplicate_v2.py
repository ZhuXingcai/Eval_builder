from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import Field, field_validator, model_validator

from env_mock_agent.facade.contracts import (
    FacadeModel,
    FacadeObjectRef,
    Identifier,
    RelativePath,
    Sha256,
)

ATTACHMENT_DUPLICATE_NORMALIZATION_VERSION: Literal["attachment-duplicate-normalization/r7-01-v1"] = (
    "attachment-duplicate-normalization/r7-01-v1"
)
ATTACHMENT_DUPLICATE_FINGERPRINT_POLICY_VERSION: Literal["attachment-duplicate-fingerprint/r7-01-v1"] = (
    "attachment-duplicate-fingerprint/r7-01-v1"
)


class AttachmentDuplicateFingerprintOutcomeV2(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    BLOCKED = "BLOCKED"


class AttachmentDuplicateFingerprintFailureCodeV2(StrEnum):
    MEDIA_UNSUPPORTED = "MEDIA_UNSUPPORTED"
    CONTENT_INSUFFICIENT = "CONTENT_INSUFFICIENT"
    MATERIAL_NOT_FOUND = "MATERIAL_NOT_FOUND"
    MATERIAL_MISMATCH = "MATERIAL_MISMATCH"
    OUTPUT_HASH_MISMATCH = "OUTPUT_HASH_MISMATCH"
    OUTPUT_PATH_VIOLATION = "OUTPUT_PATH_VIOLATION"
    SYMLINK_UNSUPPORTED = "SYMLINK_UNSUPPORTED"
    CONTENT_UNSCANNABLE = "CONTENT_UNSCANNABLE"
    SCAN_LIMIT_EXCEEDED = "SCAN_LIMIT_EXCEEDED"
    CONTAINER_UNSUPPORTED = "CONTAINER_UNSUPPORTED"


_UNSUPPORTED_FAILURE_CODES = frozenset(
    {
        AttachmentDuplicateFingerprintFailureCodeV2.MEDIA_UNSUPPORTED,
        AttachmentDuplicateFingerprintFailureCodeV2.CONTENT_INSUFFICIENT,
        AttachmentDuplicateFingerprintFailureCodeV2.CONTAINER_UNSUPPORTED,
    }
)


class AttachmentDuplicateFingerprintLimitsV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-duplicate-fingerprint-limits/v2"] = (
        "env-mock-agent/attachment-duplicate-fingerprint-limits/v2"
    )
    max_file_bytes: int = Field(ge=1, le=1_000_000_000)
    max_extracted_characters: int = Field(ge=1, le=10_000_000)
    max_inventory_members: int = Field(ge=1, le=100_000)
    max_nested_depth: int = Field(ge=1, le=16)
    max_expanded_bytes: int = Field(ge=1, le=5_000_000_000)
    max_compression_ratio_milli: int = Field(ge=1_000, le=10_000_000)
    min_token_count: int = Field(ge=1, le=100_000)
    shingle_size: int = Field(ge=1, le=16)
    fingerprint_bits: Literal[256]


class AttachmentDuplicateFingerprintRequestV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-duplicate-fingerprint-request/v2"] = (
        "env-mock-agent/attachment-duplicate-fingerprint-request/v2"
    )
    fingerprint_request_id: Identifier
    item_quality_result_ref: FacadeObjectRef
    environment_artifact_ref: FacadeObjectRef
    candidate_artifact_version_ref: FacadeObjectRef
    output_ref: FacadeObjectRef
    artifact_validation_result_ref: FacadeObjectRef
    logical_path: RelativePath
    media_type: str = Field(
        min_length=3,
        max_length=255,
        pattern=r"^[a-z0-9.+-]+/[a-z0-9.+-]+$",
    )
    content_sha256: Sha256
    size_bytes: int = Field(ge=0, le=5_000_000_000)
    limits: AttachmentDuplicateFingerprintLimitsV2
    normalization_version: Literal["attachment-duplicate-normalization/r7-01-v1"] = (
        ATTACHMENT_DUPLICATE_NORMALIZATION_VERSION
    )
    policy_version: Literal["attachment-duplicate-fingerprint/r7-01-v1"] = (
        ATTACHMENT_DUPLICATE_FINGERPRINT_POLICY_VERSION
    )
    idempotency_key: Identifier
    request_sha256: Sha256

    @model_validator(mode="after")
    def validate_request(self) -> AttachmentDuplicateFingerprintRequestV2:
        for ref, object_type, field_name in (
            (
                self.item_quality_result_ref,
                "item-quality-compilation-result",
                "item_quality_result_ref",
            ),
            (
                self.environment_artifact_ref,
                "environment-artifact",
                "environment_artifact_ref",
            ),
            (
                self.candidate_artifact_version_ref,
                "candidate-artifact-version",
                "candidate_artifact_version_ref",
            ),
            (self.output_ref, "attachment-output", "output_ref"),
            (
                self.artifact_validation_result_ref,
                "artifact-deterministic-validation-result",
                "artifact_validation_result_ref",
            ),
        ):
            _require_ref(ref, object_type, "v2", field_name)
        if self.output_ref.object_sha256 != self.content_sha256:
            raise ValueError("output_ref hash must match content_sha256")
        return self


class AttachmentDuplicateFingerprintResultV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-duplicate-fingerprint-result/v2"] = (
        "env-mock-agent/attachment-duplicate-fingerprint-result/v2"
    )
    fingerprint_result_id: Identifier
    fingerprint_request_ref: FacadeObjectRef
    environment_artifact_ref: FacadeObjectRef
    output_ref: FacadeObjectRef
    exact_content_sha256: Sha256
    outcome: AttachmentDuplicateFingerprintOutcomeV2
    similarity_fingerprint: Sha256 | None = None
    token_count: int = Field(ge=0, le=10_000_000)
    shingle_count: int = Field(ge=0, le=10_000_000)
    failure_code: AttachmentDuplicateFingerprintFailureCodeV2 | None = None
    extractor_version: str | None = Field(default=None, min_length=1, max_length=128)
    normalization_version: Literal["attachment-duplicate-normalization/r7-01-v1"] = (
        ATTACHMENT_DUPLICATE_NORMALIZATION_VERSION
    )
    policy_version: Literal["attachment-duplicate-fingerprint/r7-01-v1"] = (
        ATTACHMENT_DUPLICATE_FINGERPRINT_POLICY_VERSION
    )
    result_sha256: Sha256

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> AttachmentDuplicateFingerprintOutcomeV2:
        return _parse_enum(
            value,
            AttachmentDuplicateFingerprintOutcomeV2,
            "outcome",
        )

    @field_validator("failure_code", mode="before")
    @classmethod
    def parse_failure_code(
        cls,
        value: object,
    ) -> AttachmentDuplicateFingerprintFailureCodeV2 | None:
        if value is None:
            return None
        return _parse_enum(
            value,
            AttachmentDuplicateFingerprintFailureCodeV2,
            "failure_code",
        )

    @model_validator(mode="after")
    def validate_result(self) -> AttachmentDuplicateFingerprintResultV2:
        _require_ref(
            self.fingerprint_request_ref,
            "attachment-duplicate-fingerprint-request",
            "v2",
            "fingerprint_request_ref",
        )
        _require_ref(
            self.environment_artifact_ref,
            "environment-artifact",
            "v2",
            "environment_artifact_ref",
        )
        _require_ref(
            self.output_ref,
            "attachment-output",
            "v2",
            "output_ref",
        )
        if self.output_ref.object_sha256 != self.exact_content_sha256:
            raise ValueError("output_ref hash must match exact_content_sha256")
        if self.outcome is AttachmentDuplicateFingerprintOutcomeV2.SUPPORTED:
            if (
                self.similarity_fingerprint is None
                or self.token_count == 0
                or self.shingle_count == 0
                or self.failure_code is not None
                or self.extractor_version is None
            ):
                raise ValueError(
                    "SUPPORTED fingerprint requires fingerprint, counts, extractor, and no failure"
                )
        elif self.similarity_fingerprint is not None or self.token_count != 0 or self.shingle_count != 0:
            raise ValueError(f"{self.outcome.value} fingerprint cannot contain fingerprint or counts")
        elif self.outcome is AttachmentDuplicateFingerprintOutcomeV2.UNSUPPORTED:
            if self.failure_code not in _UNSUPPORTED_FAILURE_CODES:
                raise ValueError("UNSUPPORTED fingerprint requires an unsupported failure code")
        elif self.failure_code is None or self.failure_code in _UNSUPPORTED_FAILURE_CODES:
            raise ValueError("BLOCKED fingerprint requires a structural failure code")
        return self


class AttachmentDuplicateFingerprintFacade(Protocol):
    async def fingerprint(
        self,
        request: AttachmentDuplicateFingerprintRequestV2,
    ) -> AttachmentDuplicateFingerprintResultV2: ...


def attachment_duplicate_fingerprint_request_carried_sha256(
    request: AttachmentDuplicateFingerprintRequestV2,
) -> str:
    return _payload_sha256(
        request.model_dump(
            mode="json",
            exclude={
                "fingerprint_request_id",
                "request_sha256",
            },
            exclude_none=False,
        )
    )


def attachment_duplicate_fingerprint_request_ref(
    request: AttachmentDuplicateFingerprintRequestV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="attachment-duplicate-fingerprint-request",
        object_id=request.fingerprint_request_id,
        object_version="v2",
        object_sha256=request.request_sha256,
    )


def attachment_duplicate_fingerprint_result_carried_sha256(
    result: AttachmentDuplicateFingerprintResultV2,
) -> str:
    return _payload_sha256(
        result.model_dump(
            mode="json",
            exclude={
                "fingerprint_result_id",
                "result_sha256",
            },
            exclude_none=False,
        )
    )


def attachment_duplicate_fingerprint_result_ref(
    result: AttachmentDuplicateFingerprintResultV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="attachment-duplicate-fingerprint-result",
        object_id=result.fingerprint_result_id,
        object_version="v2",
        object_sha256=result.result_sha256,
    )


def validate_attachment_duplicate_fingerprint_request_identity(
    request: AttachmentDuplicateFingerprintRequestV2,
) -> None:
    digest = attachment_duplicate_fingerprint_request_carried_sha256(request)
    if (
        request.request_sha256 != digest
        or request.fingerprint_request_id != f"attachment-duplicate-fingerprint-request://sha256/{digest}"
    ):
        raise ValueError("attachment duplicate fingerprint request identity is stale")


def validate_attachment_duplicate_fingerprint_result_identity(
    result: AttachmentDuplicateFingerprintResultV2,
) -> None:
    digest = attachment_duplicate_fingerprint_result_carried_sha256(result)
    if (
        result.result_sha256 != digest
        or result.fingerprint_result_id != f"attachment-duplicate-fingerprint-result://sha256/{digest}"
    ):
        raise ValueError("attachment duplicate fingerprint result identity is stale")


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_ref(
    ref: FacadeObjectRef,
    expected_type: str,
    expected_version: str,
    field_name: str,
) -> None:
    if ref.object_type != expected_type or ref.object_version != expected_version:
        raise ValueError(f"{field_name} must reference {expected_type} {expected_version}")


def _parse_enum[EnumT: StrEnum](
    value: object,
    enum_type: type[EnumT],
    field_name: str,
) -> EnumT:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        return enum_type(value)
    raise TypeError(f"{field_name} must be a {enum_type.__name__}")
