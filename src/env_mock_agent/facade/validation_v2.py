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

ATTACHMENT_VALIDATION_POLICY_VERSION: Literal["attachment-validation/r5-08-v1"] = (
    "attachment-validation/r5-08-v1"
)
PROMPT_LEAKAGE_NORMALIZATION_VERSION: Literal["task-prompt-leakage-normalization/r4-04-v1"] = (
    "task-prompt-leakage-normalization/r4-04-v1"
)


class AttachmentValidationStatusV2(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class AttachmentValidationSeverityV2(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class AttachmentValidationFindingCategoryV2(StrEnum):
    STRUCTURE = "STRUCTURE"
    COVERAGE = "COVERAGE"
    EXECUTABILITY = "EXECUTABILITY"
    SECURITY = "SECURITY"
    PRIVACY = "PRIVACY"
    ANSWER_LEAKAGE = "ANSWER_LEAKAGE"
    METADATA = "METADATA"
    PACKAGE = "PACKAGE"
    TRACEABILITY = "TRACEABILITY"


class AttachmentValidationFindingCodeV2(StrEnum):
    OUTPUT_EMPTY = "OUTPUT_EMPTY"
    FORMAT_INVALID = "FORMAT_INVALID"
    CONTENT_CONTRACT_VIOLATION = "CONTENT_CONTRACT_VIOLATION"
    RENDER_CONTRACT_VIOLATION = "RENDER_CONTRACT_VIOLATION"
    SECRET_DETECTED = "SECRET_DETECTED"
    CONFIGURED_PII_DETECTED = "CONFIGURED_PII_DETECTED"
    RESTRICTED_FINGERPRINT_MATCH = "RESTRICTED_FINGERPRINT_MATCH"
    FORBIDDEN_OUTPUT_MATCH = "FORBIDDEN_OUTPUT_MATCH"
    METADATA_POLICY_VIOLATION = "METADATA_POLICY_VIOLATION"
    LINEAGE_INCOMPLETE = "LINEAGE_INCOMPLETE"


class AttachmentValidationFailureCodeV2(StrEnum):
    MATERIAL_NOT_FOUND = "MATERIAL_NOT_FOUND"
    MATERIAL_MISMATCH = "MATERIAL_MISMATCH"
    OUTPUT_MISSING = "OUTPUT_MISSING"
    OUTPUT_HASH_MISMATCH = "OUTPUT_HASH_MISMATCH"
    OUTPUT_PATH_VIOLATION = "OUTPUT_PATH_VIOLATION"
    SYMLINK_UNSUPPORTED = "SYMLINK_UNSUPPORTED"
    PACKAGE_PATH_INVALID = "PACKAGE_PATH_INVALID"
    PACKAGE_PATH_COLLISION = "PACKAGE_PATH_COLLISION"
    CONTAINER_ENCRYPTED = "CONTAINER_ENCRYPTED"
    CONTAINER_TRUNCATED = "CONTAINER_TRUNCATED"
    CONTAINER_UNSUPPORTED = "CONTAINER_UNSUPPORTED"
    CONTENT_UNSCANNABLE = "CONTENT_UNSCANNABLE"
    SCAN_LIMIT_EXCEEDED = "SCAN_LIMIT_EXCEEDED"
    INVALID_CONFIGURED_PII_RULE = "INVALID_CONFIGURED_PII_RULE"
    UNKNOWN_FINGERPRINT_NORMALIZATION = "UNKNOWN_FINGERPRINT_NORMALIZATION"
    VALIDATOR_UNAVAILABLE = "VALIDATOR_UNAVAILABLE"
    VALIDATOR_FAILED = "VALIDATOR_FAILED"
    LEGACY_FINDING_UNMAPPED = "LEGACY_FINDING_UNMAPPED"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"


class AttachmentInventoryMemberTypeV2(StrEnum):
    DIRECTORY = "DIRECTORY"
    FILE = "FILE"
    NESTED_MEMBER = "NESTED_MEMBER"


class AttachmentValidationFingerprintCategoryV2(StrEnum):
    FINAL_ANSWER = "FINAL_ANSWER"
    COMPLETED_DELIVERABLE = "COMPLETED_DELIVERABLE"
    PRIVATE_REFERENCE = "PRIVATE_REFERENCE"
    GRADER_RULE = "GRADER_RULE"
    HIDDEN_PASS_CONDITION = "HIDDEN_PASS_CONDITION"
    HIDDEN_SELECTION_SIGNAL = "HIDDEN_SELECTION_SIGNAL"
    TRAJECTORY_SPECIFIC_STEP = "TRAJECTORY_SPECIFIC_STEP"


class AttachmentValidationFingerprintMatchKindV2(StrEnum):
    NORMALIZED_FULL_TEXT = "NORMALIZED_FULL_TEXT"
    TOKEN_WINDOW = "TOKEN_WINDOW"
    PATH_COMPONENT = "PATH_COMPONENT"


class AttachmentValidationPiiRuleV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-validation-pii-rule/v2"] = (
        "env-mock-agent/attachment-validation-pii-rule/v2"
    )
    rule_id: Identifier
    pattern: str = Field(min_length=1, max_length=1000)


class AttachmentValidationFingerprintV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-validation-fingerprint/v2"] = (
        "env-mock-agent/attachment-validation-fingerprint/v2"
    )
    fingerprint_id: Identifier
    category: AttachmentValidationFingerprintCategoryV2
    match_kind: AttachmentValidationFingerprintMatchKindV2
    digest_sha256: Sha256
    token_count: int = Field(ge=1, le=20_000)
    normalization_version: str = Field(min_length=1, max_length=128)

    @field_validator("category", mode="before")
    @classmethod
    def parse_category(
        cls,
        value: object,
    ) -> AttachmentValidationFingerprintCategoryV2:
        return _parse_enum(
            value,
            AttachmentValidationFingerprintCategoryV2,
            "category",
        )

    @field_validator("match_kind", mode="before")
    @classmethod
    def parse_match_kind(
        cls,
        value: object,
    ) -> AttachmentValidationFingerprintMatchKindV2:
        return _parse_enum(
            value,
            AttachmentValidationFingerprintMatchKindV2,
            "match_kind",
        )


class AttachmentValidationScanLimitsV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-validation-scan-limits/v2"] = (
        "env-mock-agent/attachment-validation-scan-limits/v2"
    )
    max_file_bytes: int = Field(default=100_000_000, ge=1)
    max_extracted_characters: int = Field(default=2_000_000, ge=1)
    max_inventory_members: int = Field(default=20_000, ge=1)
    max_nested_depth: int = Field(default=4, ge=1, le=16)
    max_expanded_bytes: int = Field(default=500_000_000, ge=1)
    max_compression_ratio: float = Field(default=1000.0, gt=0)


class AttachmentValidationRequestV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-validation-request/v2"] = (
        "env-mock-agent/attachment-validation-request/v2"
    )
    validation_request_id: Identifier
    artifact_id: Identifier
    artifact_result_ref: FacadeObjectRef
    execution_request_ref: FacadeObjectRef
    execution_result_ref: FacadeObjectRef
    build_spec_ref: FacadeObjectRef
    producer_task_view_ref: FacadeObjectRef
    output_ref: FacadeObjectRef
    output_sha256: Sha256
    logical_path: RelativePath
    media_type: str = Field(
        min_length=3,
        max_length=255,
        pattern=r"^[a-z0-9.+-]+/[a-z0-9.+-]+$",
    )
    declared_validator_ids: tuple[Identifier, ...]
    configured_pii_rules: tuple[AttachmentValidationPiiRuleV2, ...] = ()
    leakage_reference_set_ref: FacadeObjectRef
    leakage_fingerprints: tuple[AttachmentValidationFingerprintV2, ...] = ()
    complete_leakage_categories: tuple[
        AttachmentValidationFingerprintCategoryV2,
        ...,
    ] = ()
    forbidden_output_values: tuple[str, ...] = Field(min_length=1)
    scan_limits: AttachmentValidationScanLimitsV2 = Field(default_factory=AttachmentValidationScanLimitsV2)
    policy_version: Literal["attachment-validation/r5-08-v1"] = ATTACHMENT_VALIDATION_POLICY_VERSION
    idempotency_key: Identifier
    validation_request_sha256: Sha256

    @field_validator("complete_leakage_categories", mode="before")
    @classmethod
    def parse_complete_categories(
        cls,
        value: object,
    ) -> tuple[AttachmentValidationFingerprintCategoryV2, ...]:
        if not isinstance(value, (tuple, list, set, frozenset)):
            raise TypeError("complete_leakage_categories must be a collection")
        return tuple(
            item
            if isinstance(item, AttachmentValidationFingerprintCategoryV2)
            else AttachmentValidationFingerprintCategoryV2(item)
            for item in value
        )

    @model_validator(mode="after")
    def validate_request(self) -> AttachmentValidationRequestV2:
        if (
            self.artifact_result_ref.object_type
            not in {"artifact-build-result", "candidate-artifact-version"}
            or self.artifact_result_ref.object_version != "v2"
        ):
            raise ValueError(
                "artifact_result_ref must reference an artifact build result or candidate artifact version v2"
            )
        for ref, expected_type, expected_version, field_name in (
            (
                self.execution_request_ref,
                "attachment-execution-request",
                "v2",
                "execution_request_ref",
            ),
            (
                self.execution_result_ref,
                "attachment-execution-result",
                "v2",
                "execution_result_ref",
            ),
            (
                self.build_spec_ref,
                "artifact-build-spec",
                "v2",
                "build_spec_ref",
            ),
            (
                self.producer_task_view_ref,
                "producer-task-view",
                "v2",
                "producer_task_view_ref",
            ),
            (
                self.output_ref,
                "attachment-output",
                "v2",
                "output_ref",
            ),
            (
                self.leakage_reference_set_ref,
                "prompt-leakage-reference-set",
                "v2",
                "leakage_reference_set_ref",
            ),
        ):
            _require_ref(ref, expected_type, expected_version, field_name)
        if self.output_ref.object_sha256 != self.output_sha256:
            raise ValueError("output_ref hash must match output_sha256")
        _require_sorted_unique(
            "declared validator IDs",
            self.declared_validator_ids,
        )
        _require_sorted_unique(
            "configured PII rule IDs",
            tuple(item.rule_id for item in self.configured_pii_rules),
        )
        fingerprint_keys = tuple(
            (
                item.category.value,
                item.match_kind.value,
                item.digest_sha256,
                item.token_count,
                item.fingerprint_id,
            )
            for item in self.leakage_fingerprints
        )
        _require_sorted_unique("leakage fingerprints", fingerprint_keys)
        complete_categories = tuple(item.value for item in self.complete_leakage_categories)
        _require_sorted_unique(
            "complete leakage categories",
            complete_categories,
        )
        _require_sorted_unique(
            "forbidden output values",
            self.forbidden_output_values,
        )
        return self


_FINDING_POLICY: dict[
    AttachmentValidationFindingCodeV2,
    tuple[
        AttachmentValidationSeverityV2,
        AttachmentValidationFindingCategoryV2,
        bool,
    ],
] = {
    AttachmentValidationFindingCodeV2.OUTPUT_EMPTY: (
        AttachmentValidationSeverityV2.P0,
        AttachmentValidationFindingCategoryV2.STRUCTURE,
        False,
    ),
    AttachmentValidationFindingCodeV2.FORMAT_INVALID: (
        AttachmentValidationSeverityV2.P0,
        AttachmentValidationFindingCategoryV2.STRUCTURE,
        False,
    ),
    AttachmentValidationFindingCodeV2.CONTENT_CONTRACT_VIOLATION: (
        AttachmentValidationSeverityV2.P1,
        AttachmentValidationFindingCategoryV2.COVERAGE,
        False,
    ),
    AttachmentValidationFindingCodeV2.RENDER_CONTRACT_VIOLATION: (
        AttachmentValidationSeverityV2.P1,
        AttachmentValidationFindingCategoryV2.STRUCTURE,
        False,
    ),
    AttachmentValidationFindingCodeV2.SECRET_DETECTED: (
        AttachmentValidationSeverityV2.P0,
        AttachmentValidationFindingCategoryV2.SECURITY,
        True,
    ),
    AttachmentValidationFindingCodeV2.CONFIGURED_PII_DETECTED: (
        AttachmentValidationSeverityV2.P0,
        AttachmentValidationFindingCategoryV2.PRIVACY,
        False,
    ),
    AttachmentValidationFindingCodeV2.RESTRICTED_FINGERPRINT_MATCH: (
        AttachmentValidationSeverityV2.P0,
        AttachmentValidationFindingCategoryV2.ANSWER_LEAKAGE,
        True,
    ),
    AttachmentValidationFindingCodeV2.FORBIDDEN_OUTPUT_MATCH: (
        AttachmentValidationSeverityV2.P0,
        AttachmentValidationFindingCategoryV2.ANSWER_LEAKAGE,
        True,
    ),
    AttachmentValidationFindingCodeV2.METADATA_POLICY_VIOLATION: (
        AttachmentValidationSeverityV2.P1,
        AttachmentValidationFindingCategoryV2.METADATA,
        False,
    ),
    AttachmentValidationFindingCodeV2.LINEAGE_INCOMPLETE: (
        AttachmentValidationSeverityV2.P1,
        AttachmentValidationFindingCategoryV2.TRACEABILITY,
        False,
    ),
}


def attachment_validation_finding_policy(
    code: AttachmentValidationFindingCodeV2,
) -> tuple[
    AttachmentValidationSeverityV2,
    AttachmentValidationFindingCategoryV2,
    bool,
]:
    return _FINDING_POLICY[code]


class AttachmentValidationFindingV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-validation-finding/v2"] = (
        "env-mock-agent/attachment-validation-finding/v2"
    )
    finding_id: Identifier
    artifact_id: Identifier
    subject_output_ref: FacadeObjectRef
    inventory_member_id: Identifier | None = None
    severity: AttachmentValidationSeverityV2
    category: AttachmentValidationFindingCategoryV2
    code: AttachmentValidationFindingCodeV2
    rule_ids: tuple[Identifier, ...] = ()
    matched_fingerprint_ids: tuple[Identifier, ...] = ()
    non_waivable: bool
    finding_sha256: Sha256

    @field_validator("severity", mode="before")
    @classmethod
    def parse_severity(
        cls,
        value: object,
    ) -> AttachmentValidationSeverityV2:
        return _parse_enum(value, AttachmentValidationSeverityV2, "severity")

    @field_validator("category", mode="before")
    @classmethod
    def parse_category(
        cls,
        value: object,
    ) -> AttachmentValidationFindingCategoryV2:
        return _parse_enum(
            value,
            AttachmentValidationFindingCategoryV2,
            "category",
        )

    @field_validator("code", mode="before")
    @classmethod
    def parse_code(
        cls,
        value: object,
    ) -> AttachmentValidationFindingCodeV2:
        return _parse_enum(
            value,
            AttachmentValidationFindingCodeV2,
            "code",
        )

    @model_validator(mode="after")
    def validate_finding(self) -> AttachmentValidationFindingV2:
        _require_ref(
            self.subject_output_ref,
            "attachment-output",
            "v2",
            "subject_output_ref",
        )
        expected_severity, expected_category, expected_non_waivable = attachment_validation_finding_policy(
            self.code
        )
        if (
            self.severity is not expected_severity
            or self.category is not expected_category
            or self.non_waivable is not expected_non_waivable
        ):
            raise ValueError("finding policy does not match finding code")
        _require_sorted_unique("finding rule IDs", self.rule_ids)
        _require_sorted_unique(
            "matched fingerprint IDs",
            self.matched_fingerprint_ids,
        )
        if (self.code is AttachmentValidationFindingCodeV2.RESTRICTED_FINGERPRINT_MATCH) is (
            not self.matched_fingerprint_ids
        ):
            raise ValueError("restricted fingerprint finding requires matched fingerprint IDs")
        if (
            self.code is not AttachmentValidationFindingCodeV2.RESTRICTED_FINGERPRINT_MATCH
            and self.matched_fingerprint_ids
        ):
            raise ValueError("non-fingerprint finding cannot claim matched fingerprints")
        return self


class AttachmentInventoryMemberV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-inventory-member/v2"] = (
        "env-mock-agent/attachment-inventory-member/v2"
    )
    inventory_member_id: Identifier
    artifact_id: Identifier
    output_ref: FacadeObjectRef
    normalized_path: RelativePath
    member_type: AttachmentInventoryMemberTypeV2
    media_type: str | None = Field(default=None, max_length=255)
    size_bytes: int = Field(ge=0)
    content_sha256: Sha256 | None = None
    container_ref: FacadeObjectRef | None = None
    inventory_member_sha256: Sha256

    @field_validator("member_type", mode="before")
    @classmethod
    def parse_member_type(
        cls,
        value: object,
    ) -> AttachmentInventoryMemberTypeV2:
        return _parse_enum(
            value,
            AttachmentInventoryMemberTypeV2,
            "member_type",
        )

    @model_validator(mode="after")
    def validate_member(self) -> AttachmentInventoryMemberV2:
        _require_ref(
            self.output_ref,
            "attachment-output",
            "v2",
            "output_ref",
        )
        if self.member_type is AttachmentInventoryMemberTypeV2.DIRECTORY:
            if self.size_bytes != 0 or self.content_sha256 is not None:
                raise ValueError("directory inventory member requires zero size and no hash")
            if self.container_ref is not None:
                _require_ref_type(
                    self.container_ref,
                    "attachment-output",
                    "container_ref",
                )
        elif self.member_type is AttachmentInventoryMemberTypeV2.FILE:
            if self.content_sha256 is None or self.container_ref is not None:
                raise ValueError("file inventory member requires content hash and no container")
        else:
            nested_directory = (
                self.size_bytes == 0 and self.content_sha256 is None and self.normalized_path.endswith("/")
            )
            if self.container_ref is None or (self.content_sha256 is None and not nested_directory):
                raise ValueError(
                    "nested inventory member requires container and file hash or directory marker"
                )
            _require_ref_type(
                self.container_ref,
                "attachment-output",
                "container_ref",
            )
        return self


class AttachmentValidationResultV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-validation-result/v2"] = (
        "env-mock-agent/attachment-validation-result/v2"
    )
    validation_result_id: Identifier
    validation_request_ref: FacadeObjectRef
    artifact_id: Identifier
    output_ref: FacadeObjectRef
    output_sha256: Sha256
    status: AttachmentValidationStatusV2
    executed_validator_ids: tuple[Identifier, ...]
    required_validator_ids: tuple[Identifier, ...]
    complete_leakage_categories: tuple[
        AttachmentValidationFingerprintCategoryV2,
        ...,
    ]
    findings: tuple[AttachmentValidationFindingV2, ...] = ()
    inventory_members: tuple[AttachmentInventoryMemberV2, ...] = ()
    inventory_complete: bool
    scan_complete: bool
    failure_code: AttachmentValidationFailureCodeV2 | None = None
    policy_version: Literal["attachment-validation/r5-08-v1"] = ATTACHMENT_VALIDATION_POLICY_VERSION
    validation_result_sha256: Sha256

    @field_validator("status", mode="before")
    @classmethod
    def parse_status(
        cls,
        value: object,
    ) -> AttachmentValidationStatusV2:
        return _parse_enum(value, AttachmentValidationStatusV2, "status")

    @field_validator("failure_code", mode="before")
    @classmethod
    def parse_failure_code(
        cls,
        value: object,
    ) -> AttachmentValidationFailureCodeV2 | None:
        if value is None:
            return None
        return _parse_enum(
            value,
            AttachmentValidationFailureCodeV2,
            "failure_code",
        )

    @field_validator("complete_leakage_categories", mode="before")
    @classmethod
    def parse_complete_categories(
        cls,
        value: object,
    ) -> tuple[AttachmentValidationFingerprintCategoryV2, ...]:
        if not isinstance(value, (tuple, list, set, frozenset)):
            raise TypeError("complete_leakage_categories must be a collection")
        return tuple(
            item
            if isinstance(item, AttachmentValidationFingerprintCategoryV2)
            else AttachmentValidationFingerprintCategoryV2(item)
            for item in value
        )

    @model_validator(mode="after")
    def validate_result(self) -> AttachmentValidationResultV2:
        _require_ref(
            self.validation_request_ref,
            "attachment-validation-request",
            "v2",
            "validation_request_ref",
        )
        _require_ref(
            self.output_ref,
            "attachment-output",
            "v2",
            "output_ref",
        )
        if self.output_ref.object_sha256 != self.output_sha256:
            raise ValueError("validation output ref hash must match output_sha256")
        _require_sorted_unique(
            "executed validator IDs",
            self.executed_validator_ids,
        )
        _require_sorted_unique(
            "required validator IDs",
            self.required_validator_ids,
        )
        complete_categories = tuple(item.value for item in self.complete_leakage_categories)
        _require_sorted_unique(
            "complete leakage categories",
            complete_categories,
        )
        finding_ids = tuple(item.finding_id for item in self.findings)
        _require_sorted_unique("validation finding IDs", finding_ids)
        member_keys = tuple(_inventory_member_key(item) for item in self.inventory_members)
        _require_sorted_unique("inventory members", member_keys)
        member_ids = {item.inventory_member_id for item in self.inventory_members}
        for finding in self.findings:
            if finding.artifact_id != self.artifact_id or finding.subject_output_ref != self.output_ref:
                raise ValueError("validation finding subject does not match result")
            if finding.inventory_member_id is not None and finding.inventory_member_id not in member_ids:
                raise ValueError("validation finding references unknown inventory member")
        for member in self.inventory_members:
            if member.artifact_id != self.artifact_id or member.output_ref != self.output_ref:
                raise ValueError("inventory member subject does not match result")

        validators_complete = self.executed_validator_ids == self.required_validator_ids
        if self.status is AttachmentValidationStatusV2.PASSED:
            if (
                not validators_complete
                or self.findings
                or not self.inventory_complete
                or not self.scan_complete
                or self.failure_code is not None
            ):
                raise ValueError(
                    "PASSED validation requires complete validators/scan/inventory and no findings"
                )
        elif self.status is AttachmentValidationStatusV2.FAILED:
            if (
                not validators_complete
                or not self.findings
                or not self.inventory_complete
                or not self.scan_complete
                or self.failure_code is not None
            ):
                raise ValueError("FAILED validation requires complete validators/scan/inventory and findings")
        elif self.failure_code is None or (
            self.inventory_complete and self.scan_complete and validators_complete
        ):
            raise ValueError("BLOCKED validation requires failure code and incomplete validation")
        return self


class AttachmentValidationFacade(Protocol):
    async def validate(
        self,
        request: AttachmentValidationRequestV2,
    ) -> AttachmentValidationResultV2: ...


def attachment_validation_request_carried_sha256(
    request: AttachmentValidationRequestV2,
) -> str:
    return _payload_sha256(
        request.model_dump(
            mode="json",
            exclude={
                "validation_request_id",
                "validation_request_sha256",
            },
            exclude_none=False,
        )
    )


def attachment_validation_request_ref(
    request: AttachmentValidationRequestV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="attachment-validation-request",
        object_id=request.validation_request_id,
        object_version="v2",
        object_sha256=request.validation_request_sha256,
    )


def attachment_validation_finding_carried_sha256(
    finding: AttachmentValidationFindingV2,
) -> str:
    return _payload_sha256(
        finding.model_dump(
            mode="json",
            exclude={"finding_id", "finding_sha256"},
            exclude_none=False,
        )
    )


def attachment_validation_finding_ref(
    finding: AttachmentValidationFindingV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="attachment-validation-finding",
        object_id=finding.finding_id,
        object_version="v2",
        object_sha256=finding.finding_sha256,
    )


def attachment_inventory_member_carried_sha256(
    member: AttachmentInventoryMemberV2,
) -> str:
    return _payload_sha256(
        member.model_dump(
            mode="json",
            exclude={
                "inventory_member_id",
                "inventory_member_sha256",
            },
            exclude_none=False,
        )
    )


def attachment_inventory_member_ref(
    member: AttachmentInventoryMemberV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="attachment-inventory-member",
        object_id=member.inventory_member_id,
        object_version="v2",
        object_sha256=member.inventory_member_sha256,
    )


def attachment_validation_result_carried_sha256(
    result: AttachmentValidationResultV2,
) -> str:
    return _payload_sha256(
        result.model_dump(
            mode="json",
            exclude={
                "validation_result_id",
                "validation_result_sha256",
            },
            exclude_none=False,
        )
    )


def attachment_validation_result_ref(
    result: AttachmentValidationResultV2,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type="attachment-validation-result",
        object_id=result.validation_result_id,
        object_version="v2",
        object_sha256=result.validation_result_sha256,
    )


def validate_attachment_validation_request_identity(
    request: AttachmentValidationRequestV2,
) -> None:
    digest = attachment_validation_request_carried_sha256(request)
    if (
        request.validation_request_sha256 != digest
        or request.validation_request_id != f"attachment-validation-request://sha256/{digest}"
    ):
        raise ValueError("attachment validation request identity is stale")


def validate_attachment_validation_finding_identity(
    finding: AttachmentValidationFindingV2,
) -> None:
    digest = attachment_validation_finding_carried_sha256(finding)
    if (
        finding.finding_sha256 != digest
        or finding.finding_id != f"attachment-validation-finding://sha256/{digest}"
    ):
        raise ValueError("attachment validation finding identity is stale")


def validate_attachment_inventory_member_identity(
    member: AttachmentInventoryMemberV2,
) -> None:
    digest = attachment_inventory_member_carried_sha256(member)
    if (
        member.inventory_member_sha256 != digest
        or member.inventory_member_id != f"attachment-inventory-member://sha256/{digest}"
    ):
        raise ValueError("attachment inventory member identity is stale")


def validate_attachment_validation_result_identity(
    result: AttachmentValidationResultV2,
) -> None:
    for finding in result.findings:
        validate_attachment_validation_finding_identity(finding)
    for member in result.inventory_members:
        validate_attachment_inventory_member_identity(member)
    digest = attachment_validation_result_carried_sha256(result)
    if (
        result.validation_result_sha256 != digest
        or result.validation_result_id != f"attachment-validation-result://sha256/{digest}"
    ):
        raise ValueError("attachment validation result identity is stale")


def _inventory_member_key(
    member: AttachmentInventoryMemberV2,
) -> tuple[str, str, str]:
    return (
        member.container_ref.object_id if member.container_ref is not None else "",
        member.normalized_path,
        member.member_type.value,
    )


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


def _require_ref_type(
    ref: FacadeObjectRef,
    expected_type: str,
    field_name: str,
) -> None:
    if ref.object_type != expected_type:
        raise ValueError(f"{field_name} must reference {expected_type}")


def _require_sorted_unique(
    label: str,
    values: tuple[object, ...],
) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    if values != tuple(sorted(values, key=repr)):
        raise ValueError(f"{label} must be sorted")


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
