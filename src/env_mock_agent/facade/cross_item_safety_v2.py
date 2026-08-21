from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import Field, field_validator, model_validator

from env_mock_agent.facade.contracts import (
    FacadeModel,
    FacadeObjectRef,
    Identifier,
    RelativePath,
    Sha256,
)
from env_mock_agent.facade.validation_v2 import (
    PROMPT_LEAKAGE_NORMALIZATION_VERSION,
    AttachmentValidationFingerprintV2,
)

ATTACHMENT_CROSS_ITEM_SAFETY_POLICY_VERSION: Literal["attachment-cross-item-safety/r7-02-v1"] = (
    "attachment-cross-item-safety/r7-02-v1"
)


class AttachmentCrossItemSafetyScanStatusV2(StrEnum):
    PASSED = "PASSED"
    BLOCKED = "BLOCKED"


class AttachmentCrossItemSafetyScanFailureCodeV2(StrEnum):
    MATERIAL_NOT_FOUND = "MATERIAL_NOT_FOUND"
    MATERIAL_MISMATCH = "MATERIAL_MISMATCH"
    OUTPUT_HASH_MISMATCH = "OUTPUT_HASH_MISMATCH"
    OUTPUT_PATH_VIOLATION = "OUTPUT_PATH_VIOLATION"
    SYMLINK_UNSUPPORTED = "SYMLINK_UNSUPPORTED"
    CONTAINER_ENCRYPTED = "CONTAINER_ENCRYPTED"
    CONTAINER_TRUNCATED = "CONTAINER_TRUNCATED"
    CONTAINER_UNSUPPORTED = "CONTAINER_UNSUPPORTED"
    CONTENT_UNSCANNABLE = "CONTENT_UNSCANNABLE"
    SCAN_LIMIT_EXCEEDED = "SCAN_LIMIT_EXCEEDED"
    UNKNOWN_FINGERPRINT_NORMALIZATION = "UNKNOWN_FINGERPRINT_NORMALIZATION"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"


class AttachmentCrossItemSafetyScanLimitsV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-cross-item-safety-scan-limits/v2"] = (
        "env-mock-agent/attachment-cross-item-safety-scan-limits/v2"
    )
    max_file_bytes: int = Field(ge=1, le=1_000_000_000)
    max_extracted_characters: int = Field(ge=1, le=10_000_000)
    max_inventory_members: int = Field(ge=1, le=100_000)
    max_nested_depth: int = Field(ge=1, le=16)
    max_expanded_bytes: int = Field(ge=1, le=5_000_000_000)
    max_compression_ratio_milli: int = Field(ge=1_000, le=10_000_000)
    max_fingerprint_count: int = Field(ge=0, le=1_000_000)
    max_match_count: int = Field(ge=0, le=1_000_000)


class AttachmentCrossItemSafetyScanRequestV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-cross-item-safety-scan-request/v2"] = (
        "env-mock-agent/attachment-cross-item-safety-scan-request/v2"
    )
    scan_request_id: Identifier
    target_item_id: Identifier
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
    foreign_reference_set_refs: tuple[FacadeObjectRef, ...] = Field(
        min_length=1,
        max_length=100_000,
    )
    foreign_fingerprints: tuple[AttachmentValidationFingerprintV2, ...] = Field(
        max_length=1_000_000,
    )
    limits: AttachmentCrossItemSafetyScanLimitsV2
    normalization_version: Literal["task-prompt-leakage-normalization/r4-04-v1"] = (
        PROMPT_LEAKAGE_NORMALIZATION_VERSION
    )
    policy_version: Literal["attachment-cross-item-safety/r7-02-v1"] = (
        ATTACHMENT_CROSS_ITEM_SAFETY_POLICY_VERSION
    )
    idempotency_key: Identifier
    request_sha256: Sha256

    @model_validator(mode="after")
    def validate_request(self) -> AttachmentCrossItemSafetyScanRequestV2:
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
        _require_sorted_unique_refs(
            "foreign reference sets",
            self.foreign_reference_set_refs,
        )
        for ref in self.foreign_reference_set_refs:
            _require_ref(
                ref,
                "prompt-leakage-reference-set",
                "v2",
                "foreign_reference_set_refs",
            )
        fingerprint_keys = tuple(
            (
                value.category.value,
                value.match_kind.value,
                value.digest_sha256,
                value.token_count,
                value.fingerprint_id,
            )
            for value in self.foreign_fingerprints
        )
        _require_sorted_unique("foreign fingerprints", fingerprint_keys)
        fingerprint_ids = tuple(value.fingerprint_id for value in self.foreign_fingerprints)
        if len(fingerprint_ids) != len(set(fingerprint_ids)):
            raise ValueError("foreign fingerprint IDs must be unique")
        if len(self.foreign_fingerprints) > self.limits.max_fingerprint_count:
            raise ValueError("foreign fingerprint count exceeds request limit")
        if any(
            value.normalization_version != self.normalization_version for value in self.foreign_fingerprints
        ):
            raise ValueError("foreign fingerprint normalization is mismatched")
        return self


class AttachmentCrossItemSafetyScanResultV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-cross-item-safety-scan-result/v2"] = (
        "env-mock-agent/attachment-cross-item-safety-scan-result/v2"
    )
    scan_result_id: Identifier
    scan_request_ref: FacadeObjectRef
    environment_artifact_ref: FacadeObjectRef
    output_ref: FacadeObjectRef
    output_sha256: Sha256
    status: AttachmentCrossItemSafetyScanStatusV2
    matched_fingerprint_ids: tuple[Identifier, ...] = Field(
        max_length=1_000_000,
    )
    scanned_member_count: int = Field(ge=0, le=100_000)
    scan_complete: bool
    failure_code: AttachmentCrossItemSafetyScanFailureCodeV2 | None = None
    extractor_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    normalization_version: Literal["task-prompt-leakage-normalization/r4-04-v1"] = (
        PROMPT_LEAKAGE_NORMALIZATION_VERSION
    )
    policy_version: Literal["attachment-cross-item-safety/r7-02-v1"] = (
        ATTACHMENT_CROSS_ITEM_SAFETY_POLICY_VERSION
    )
    result_sha256: Sha256

    @field_validator("status", mode="before")
    @classmethod
    def parse_status(
        cls,
        value: object,
    ) -> AttachmentCrossItemSafetyScanStatusV2:
        return _parse_enum(
            value,
            AttachmentCrossItemSafetyScanStatusV2,
            "status",
        )

    @field_validator("failure_code", mode="before")
    @classmethod
    def parse_failure_code(
        cls,
        value: object,
    ) -> AttachmentCrossItemSafetyScanFailureCodeV2 | None:
        if value is None:
            return None
        return _parse_enum(
            value,
            AttachmentCrossItemSafetyScanFailureCodeV2,
            "failure_code",
        )

    @model_validator(mode="after")
    def validate_result(self) -> AttachmentCrossItemSafetyScanResultV2:
        _require_ref(
            self.scan_request_ref,
            "attachment-cross-item-safety-scan-request",
            "v2",
            "scan_request_ref",
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
        if self.output_ref.object_sha256 != self.output_sha256:
            raise ValueError("output_ref hash must match output_sha256")
        _require_sorted_unique(
            "matched fingerprint IDs",
            self.matched_fingerprint_ids,
        )
        if self.status is AttachmentCrossItemSafetyScanStatusV2.PASSED:
            if not self.scan_complete or self.failure_code is not None or self.extractor_version is None:
                raise ValueError("PASSED scan requires complete scan, extractor, and no failure")
        elif (
            self.scan_complete
            or self.matched_fingerprint_ids
            or self.scanned_member_count != 0
            or self.failure_code is None
            or self.extractor_version is not None
        ):
            raise ValueError("BLOCKED scan requires one failure and no partial scan evidence")
        return self


class AttachmentCrossItemSafetyScanFacade(Protocol):
    async def scan(
        self,
        request: AttachmentCrossItemSafetyScanRequestV2,
    ) -> AttachmentCrossItemSafetyScanResultV2: ...


def attachment_cross_item_safety_scan_request_carried_sha256(
    request: AttachmentCrossItemSafetyScanRequestV2,
) -> str:
    return _payload_sha256(
        request.model_dump(
            mode="json",
            exclude={
                "scan_request_id",
                "request_sha256",
            },
            exclude_none=False,
        )
    )


def attachment_cross_item_safety_scan_request_ref(
    request: AttachmentCrossItemSafetyScanRequestV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="attachment-cross-item-safety-scan-request",
        object_id=request.scan_request_id,
        object_version="v2",
        object_sha256=request.request_sha256,
    )


def attachment_cross_item_safety_scan_result_carried_sha256(
    result: AttachmentCrossItemSafetyScanResultV2,
) -> str:
    return _payload_sha256(
        result.model_dump(
            mode="json",
            exclude={
                "scan_result_id",
                "result_sha256",
            },
            exclude_none=False,
        )
    )


def attachment_cross_item_safety_scan_result_ref(
    result: AttachmentCrossItemSafetyScanResultV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="attachment-cross-item-safety-scan-result",
        object_id=result.scan_result_id,
        object_version="v2",
        object_sha256=result.result_sha256,
    )


def validate_attachment_cross_item_safety_scan_request_identity(
    request: AttachmentCrossItemSafetyScanRequestV2,
) -> None:
    digest = attachment_cross_item_safety_scan_request_carried_sha256(request)
    if (
        request.request_sha256 != digest
        or request.scan_request_id != f"attachment-cross-item-safety-scan-request://sha256/{digest}"
    ):
        raise ValueError("attachment cross-item safety scan request identity is stale")


def validate_attachment_cross_item_safety_scan_result_identity(
    result: AttachmentCrossItemSafetyScanResultV2,
) -> None:
    digest = attachment_cross_item_safety_scan_result_carried_sha256(result)
    if (
        result.result_sha256 != digest
        or result.scan_result_id != f"attachment-cross-item-safety-scan-result://sha256/{digest}"
    ):
        raise ValueError("attachment cross-item safety scan result identity is stale")


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


def _ref_key(ref: FacadeObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _require_sorted_unique_refs(
    label: str,
    values: tuple[FacadeObjectRef, ...],
) -> None:
    keys = tuple(_ref_key(value) for value in values)
    _require_sorted_unique(label, keys)


def _require_sorted_unique(
    label: str,
    values: tuple[Any, ...],
) -> None:
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{label} must be sorted and unique")


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
