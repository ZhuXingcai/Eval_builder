from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import field_validator, model_validator

from env_mock_agent.facade import (
    AttachmentInventoryMemberV2,
    AttachmentValidationFailureCodeV2,
    AttachmentValidationFindingCodeV2,
    AttachmentValidationResultV2,
    AttachmentValidationStatusV2,
    FacadeObjectRef,
    attachment_inventory_member_ref,
    attachment_validation_finding_policy,
    attachment_validation_finding_ref,
    attachment_validation_result_ref,
    validate_attachment_inventory_member_identity,
    validate_attachment_validation_result_identity,
)
from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.quality import Severity
from eval_factory.contracts.safety import PackageInventoryMember

DETERMINISTIC_VALIDATION_POLICY_VERSION: Literal["deterministic-validation/r5-08-v1"] = (
    "deterministic-validation/r5-08-v1"
)


class DeterministicValidationFindingCategoryV2(StrEnum):
    STRUCTURE = "STRUCTURE"
    COVERAGE = "COVERAGE"
    EXECUTABILITY = "EXECUTABILITY"
    SECURITY = "SECURITY"
    PRIVACY = "PRIVACY"
    ANSWER_LEAKAGE = "ANSWER_LEAKAGE"
    METADATA = "METADATA"
    PACKAGE = "PACKAGE"
    TRACEABILITY = "TRACEABILITY"


class DeterministicValidationFindingCodeV2(StrEnum):
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


class ArtifactDeterministicValidationOutcomeV2(StrEnum):
    PASSED = "PASSED"
    REQUIRES_REPAIR = "REQUIRES_REPAIR"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"


class DeterministicItemValidationOutcomeV2(StrEnum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PASSED = "PASSED"
    UPSTREAM_INCOMPLETE = "UPSTREAM_INCOMPLETE"
    REQUIRES_REPAIR = "REQUIRES_REPAIR"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"


class DeterministicValidationFindingV2(ContractModelV2):
    schema_version: Literal["eval-factory/deterministic-validation-finding/v2"] = (
        "eval-factory/deterministic-validation-finding/v2"
    )
    finding_id: Identifier
    artifact_build_result_ref: ObjectRef
    output_ref: ObjectRef
    facade_validation_result_ref: ObjectRef
    facade_finding_ref: ObjectRef
    inventory_member_ref: ObjectRef | None = None
    severity: Severity
    category: DeterministicValidationFindingCategoryV2
    code: DeterministicValidationFindingCodeV2
    rule_ids: tuple[Identifier, ...] = ()
    matched_fingerprint_ids: tuple[Identifier, ...] = ()
    status: Literal["OPEN"] = "OPEN"
    non_waivable: bool
    policy_version: Literal["deterministic-validation/r5-08-v1"] = DETERMINISTIC_VALIDATION_POLICY_VERSION
    finding_sha256: Sha256
    audit: ContractAudit

    @field_validator("severity", mode="before")
    @classmethod
    def parse_severity(cls, value: object) -> Severity:
        return _parse_enum(value, Severity, "severity")

    @field_validator("category", mode="before")
    @classmethod
    def parse_category(
        cls,
        value: object,
    ) -> DeterministicValidationFindingCategoryV2:
        return _parse_enum(
            value,
            DeterministicValidationFindingCategoryV2,
            "category",
        )

    @field_validator("code", mode="before")
    @classmethod
    def parse_code(
        cls,
        value: object,
    ) -> DeterministicValidationFindingCodeV2:
        return _parse_enum(
            value,
            DeterministicValidationFindingCodeV2,
            "code",
        )

    @model_validator(mode="after")
    def validate_finding(self) -> DeterministicValidationFindingV2:
        _require_ref_one_of(
            self.artifact_build_result_ref,
            ("artifact-build-result", "candidate-artifact-version"),
            "v2",
            "artifact_build_result_ref",
        )
        for ref, expected_type, field_name in (
            (self.output_ref, "attachment-output", "output_ref"),
            (
                self.facade_validation_result_ref,
                "attachment-validation-result",
                "facade_validation_result_ref",
            ),
            (
                self.facade_finding_ref,
                "attachment-validation-finding",
                "facade_finding_ref",
            ),
        ):
            _require_ref(ref, expected_type, "v2", field_name)
        if self.inventory_member_ref is not None:
            _require_ref(
                self.inventory_member_ref,
                "attachment-inventory-member",
                "v2",
                "inventory_member_ref",
            )
        facade_code = AttachmentValidationFindingCodeV2(self.code.value)
        facade_severity, facade_category, non_waivable = attachment_validation_finding_policy(facade_code)
        if (
            self.severity.value != facade_severity.value
            or self.category.value != facade_category.value
            or self.non_waivable is not non_waivable
        ):
            raise ValueError("deterministic finding policy does not match facade code")
        _require_sorted_unique("finding rule IDs", self.rule_ids)
        _require_sorted_unique(
            "matched fingerprint IDs",
            self.matched_fingerprint_ids,
        )
        expected_audit_refs = [
            self.artifact_build_result_ref,
            self.output_ref,
            self.facade_validation_result_ref,
            self.facade_finding_ref,
        ]
        if self.inventory_member_ref is not None:
            expected_audit_refs.append(self.inventory_member_ref)
        _validate_audit(
            self.audit,
            tuple(sorted(expected_audit_refs, key=_ref_key)),
            "deterministic finding",
        )
        return self


class ArtifactDeterministicValidationResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/artifact-deterministic-validation-result/v2"] = (
        "eval-factory/artifact-deterministic-validation-result/v2"
    )
    artifact_validation_result_id: Identifier
    attachment_reconstruction_result_ref: ObjectRef
    artifact_build_result_ref: ObjectRef
    build_spec_ref: ObjectRef
    output_ref: ObjectRef
    output_sha256: Sha256
    facade_validation_request_ref: ObjectRef
    facade_validation_result: AttachmentValidationResultV2
    outcome: ArtifactDeterministicValidationOutcomeV2
    findings: tuple[DeterministicValidationFindingV2, ...] = ()
    finding_refs: tuple[ObjectRef, ...] = ()
    inventory_members: tuple[AttachmentInventoryMemberV2, ...] = ()
    policy_version: Literal["deterministic-validation/r5-08-v1"] = DETERMINISTIC_VALIDATION_POLICY_VERSION
    artifact_validation_result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> ArtifactDeterministicValidationOutcomeV2:
        return _parse_enum(
            value,
            ArtifactDeterministicValidationOutcomeV2,
            "outcome",
        )

    @model_validator(mode="after")
    def validate_result(self) -> ArtifactDeterministicValidationResultV2:
        _require_ref_one_of(
            self.attachment_reconstruction_result_ref,
            (
                "attachment-reconstruction-result",
                "attachment-candidate-revision",
            ),
            "v2",
            "attachment_reconstruction_result_ref",
        )
        _require_ref_one_of(
            self.artifact_build_result_ref,
            ("artifact-build-result", "candidate-artifact-version"),
            "v2",
            "artifact_build_result_ref",
        )
        for ref, expected_type, field_name in (
            (self.build_spec_ref, "artifact-build-spec", "build_spec_ref"),
            (self.output_ref, "attachment-output", "output_ref"),
            (
                self.facade_validation_request_ref,
                "attachment-validation-request",
                "facade_validation_request_ref",
            ),
        ):
            _require_ref(ref, expected_type, "v2", field_name)
        if self.output_ref.object_sha256 != self.output_sha256:
            raise ValueError("artifact validation output ref/hash mismatch")
        validate_attachment_validation_result_identity(self.facade_validation_result)
        facade_result_ref = _object_ref_from_facade(
            attachment_validation_result_ref(self.facade_validation_result)
        )
        if (
            _object_ref_from_facade(self.facade_validation_result.validation_request_ref)
            != self.facade_validation_request_ref
            or self.facade_validation_result.output_ref.object_sha256 != self.output_sha256
            or _object_ref_from_facade(self.facade_validation_result.output_ref) != self.output_ref
        ):
            raise ValueError("artifact validation facade result does not match subject")
        artifact_ids = {item.artifact_id for item in self.inventory_members} | {
            self.facade_validation_result.artifact_id
        }
        if len(artifact_ids) != 1:
            raise ValueError("artifact validation inventory has multiple artifact IDs")
        expected_inventory = self.facade_validation_result.inventory_members
        if self.inventory_members != expected_inventory:
            raise ValueError("artifact validation inventory must equal facade inventory")
        for member in self.inventory_members:
            validate_attachment_inventory_member_identity(member)
        expected_finding_refs = tuple(deterministic_validation_finding_ref(item) for item in self.findings)
        if self.finding_refs != expected_finding_refs:
            raise ValueError("artifact validation finding refs must match nested findings")
        facade_finding_refs = tuple(
            sorted(
                (
                    _object_ref_from_facade(attachment_validation_finding_ref(item))
                    for item in self.facade_validation_result.findings
                ),
                key=_ref_key,
            )
        )
        observed_facade_refs = tuple(
            sorted(
                (item.facade_finding_ref for item in self.findings),
                key=_ref_key,
            )
        )
        if observed_facade_refs != facade_finding_refs:
            raise ValueError("factory findings must exactly map facade findings")
        if any(
            item.artifact_build_result_ref != self.artifact_build_result_ref
            or item.output_ref != self.output_ref
            or item.facade_validation_result_ref != facade_result_ref
            for item in self.findings
        ):
            raise ValueError("artifact validation findings do not match exact subject")
        expected_outcome = _artifact_outcome(
            self.facade_validation_result,
            self.findings,
        )
        if self.outcome is not expected_outcome:
            raise ValueError("artifact deterministic validation outcome is not exact")
        expected_audit_refs = tuple(
            sorted(
                (
                    self.attachment_reconstruction_result_ref,
                    self.artifact_build_result_ref,
                    self.build_spec_ref,
                    self.output_ref,
                    self.facade_validation_request_ref,
                    facade_result_ref,
                    *self.finding_refs,
                ),
                key=_ref_key,
            )
        )
        _validate_audit(
            self.audit,
            expected_audit_refs,
            "artifact deterministic validation",
        )
        return self


class CandidatePackageInventoryEntryV2(ContractModelV2):
    schema_version: Literal["eval-factory/candidate-package-inventory-entry/v2"] = (
        "eval-factory/candidate-package-inventory-entry/v2"
    )
    inventory_member: PackageInventoryMember
    inventory_member_ref: ObjectRef
    container_ref: ObjectRef | None = None
    artifact_build_result_ref: ObjectRef
    artifact_validation_result_ref: ObjectRef
    output_ref: ObjectRef
    build_spec_ref: ObjectRef
    derivation_root_refs: tuple[ObjectRef, ...]
    entry_sha256: Sha256

    @model_validator(mode="after")
    def validate_entry(self) -> CandidatePackageInventoryEntryV2:
        _require_ref_one_of(
            self.artifact_build_result_ref,
            ("artifact-build-result", "candidate-artifact-version"),
            "v2",
            "artifact_build_result_ref",
        )
        for ref, expected_type, field_name in (
            (
                self.inventory_member_ref,
                "attachment-inventory-member",
                "inventory_member_ref",
            ),
            (
                self.artifact_validation_result_ref,
                "artifact-deterministic-validation-result",
                "artifact_validation_result_ref",
            ),
            (self.output_ref, "attachment-output", "output_ref"),
            (self.build_spec_ref, "artifact-build-spec", "build_spec_ref"),
        ):
            _require_ref(ref, expected_type, "v2", field_name)
        if self.container_ref is not None:
            _require_ref(
                self.container_ref,
                "attachment-output",
                "v2",
                "container_ref",
            )
        if self.inventory_member.container_ref != (
            self.container_ref.object_id if self.container_ref is not None else None
        ):
            raise ValueError("candidate inventory container ref does not match frozen member")
        _require_sorted_unique_refs(
            "candidate derivation roots",
            self.derivation_root_refs,
        )
        if not self.derivation_root_refs:
            raise ValueError("candidate inventory entry requires derivation roots")
        return self


class CandidatePackageInventoryV2(ContractModelV2):
    schema_version: Literal["eval-factory/candidate-package-inventory/v2"] = (
        "eval-factory/candidate-package-inventory/v2"
    )
    candidate_inventory_id: Identifier
    attachment_reconstruction_result_ref: ObjectRef
    artifact_validation_result_refs: tuple[ObjectRef, ...]
    entries: tuple[CandidatePackageInventoryEntryV2, ...]
    exact_candidate_set_verified: Literal[True] = True
    final_package: Literal[False] = False
    provenance_manifest_ref: None = None
    package_sha256: None = None
    input_state_only: None = None
    policy_version: Literal["deterministic-validation/r5-08-v1"] = DETERMINISTIC_VALIDATION_POLICY_VERSION
    candidate_inventory_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_inventory(self) -> CandidatePackageInventoryV2:
        _require_ref_one_of(
            self.attachment_reconstruction_result_ref,
            (
                "attachment-reconstruction-result",
                "attachment-candidate-revision",
            ),
            "v2",
            "attachment_reconstruction_result_ref",
        )
        if len(self.artifact_validation_result_refs) != len(set(self.artifact_validation_result_refs)):
            raise ValueError("artifact validation result refs must be unique")
        entry_keys = tuple(_candidate_entry_key(item) for item in self.entries)
        _require_sorted_unique("candidate inventory entries", entry_keys)
        expected_validation_refs = {item.artifact_validation_result_ref for item in self.entries}
        if set(self.artifact_validation_result_refs) != expected_validation_refs:
            raise ValueError("candidate inventory validation refs must equal entry owners")
        expected_audit_refs = tuple(
            sorted(
                (
                    self.attachment_reconstruction_result_ref,
                    *self.artifact_validation_result_refs,
                    *(candidate_package_inventory_entry_ref(item) for item in self.entries),
                ),
                key=_ref_key,
            )
        )
        _validate_audit(
            self.audit,
            expected_audit_refs,
            "candidate package inventory",
        )
        return self


class DeterministicItemValidationResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/deterministic-item-validation-result/v2"] = (
        "eval-factory/deterministic-item-validation-result/v2"
    )
    deterministic_item_validation_result_id: Identifier
    attachment_reconstruction_result_ref: ObjectRef
    producer_task_view_ref: ObjectRef
    leakage_reference_set_ref: ObjectRef
    configured_pii_rules_sha256: Sha256 = "0" * 64
    artifact_validation_results: tuple[
        ArtifactDeterministicValidationResultV2,
        ...,
    ] = ()
    artifact_validation_result_refs: tuple[ObjectRef, ...] = ()
    finding_refs: tuple[ObjectRef, ...] = ()
    candidate_inventory: CandidatePackageInventoryV2 | None = None
    candidate_inventory_ref: ObjectRef | None = None
    candidate_inventory_failure_code: AttachmentValidationFailureCodeV2 | None = None
    outcome: DeterministicItemValidationOutcomeV2
    validated_artifact_ids: tuple[Identifier, ...] = ()
    repair_required_artifact_ids: tuple[Identifier, ...] = ()
    rejected_artifact_ids: tuple[Identifier, ...] = ()
    validation_blocked_artifact_ids: tuple[Identifier, ...] = ()
    upstream_incomplete_artifact_ids: tuple[Identifier, ...] = ()
    accepted_artifact_refs: tuple[ObjectRef, ...] = ()
    environment_spec_ref: None = None
    provenance_manifest_ref: None = None
    quality_report_ref: None = None
    package_sha256: None = None
    input_state_only: None = None
    policy_version: Literal["deterministic-validation/r5-08-v1"] = DETERMINISTIC_VALIDATION_POLICY_VERSION
    deterministic_item_validation_result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> DeterministicItemValidationOutcomeV2:
        return _parse_enum(
            value,
            DeterministicItemValidationOutcomeV2,
            "outcome",
        )

    @field_validator("candidate_inventory_failure_code", mode="before")
    @classmethod
    def parse_inventory_failure_code(
        cls,
        value: object,
    ) -> AttachmentValidationFailureCodeV2 | None:
        if value is None:
            return None
        return _parse_enum(
            value,
            AttachmentValidationFailureCodeV2,
            "candidate_inventory_failure_code",
        )

    @model_validator(mode="after")
    def validate_result(self) -> DeterministicItemValidationResultV2:
        _require_ref_one_of(
            self.attachment_reconstruction_result_ref,
            (
                "attachment-reconstruction-result",
                "attachment-candidate-revision",
            ),
            "v2",
            "attachment_reconstruction_result_ref",
        )
        for ref, expected_type, field_name in (
            (
                self.producer_task_view_ref,
                "producer-task-view",
                "producer_task_view_ref",
            ),
            (
                self.leakage_reference_set_ref,
                "prompt-leakage-reference-set",
                "leakage_reference_set_ref",
            ),
        ):
            _require_ref(ref, expected_type, "v2", field_name)
        artifact_ids = tuple(
            item.facade_validation_result.artifact_id for item in self.artifact_validation_results
        )
        _require_sorted_unique("artifact validation IDs", artifact_ids)
        expected_result_refs = tuple(
            artifact_deterministic_validation_result_ref(item) for item in self.artifact_validation_results
        )
        if self.artifact_validation_result_refs != expected_result_refs:
            raise ValueError("artifact validation refs must match nested results")
        expected_finding_refs = tuple(
            sorted(
                (ref for item in self.artifact_validation_results for ref in item.finding_refs),
                key=_ref_key,
            )
        )
        if self.finding_refs != expected_finding_refs:
            raise ValueError("item deterministic finding refs must match artifact results")
        expected_validated = _artifact_ids_for_outcome(
            self.artifact_validation_results,
            {ArtifactDeterministicValidationOutcomeV2.PASSED},
        )
        expected_repair = _artifact_ids_for_outcome(
            self.artifact_validation_results,
            {ArtifactDeterministicValidationOutcomeV2.REQUIRES_REPAIR},
        )
        expected_rejected = _artifact_ids_for_outcome(
            self.artifact_validation_results,
            {ArtifactDeterministicValidationOutcomeV2.REJECTED},
        )
        expected_blocked = _artifact_ids_for_outcome(
            self.artifact_validation_results,
            {ArtifactDeterministicValidationOutcomeV2.BLOCKED},
        )
        if (
            self.validated_artifact_ids != expected_validated
            or self.repair_required_artifact_ids != expected_repair
            or self.rejected_artifact_ids != expected_rejected
            or self.validation_blocked_artifact_ids != expected_blocked
        ):
            raise ValueError("item deterministic artifact inventories are not exact")
        _require_sorted_unique(
            "upstream incomplete artifact IDs",
            self.upstream_incomplete_artifact_ids,
        )
        classified_ids = set(artifact_ids)
        if classified_ids & set(self.upstream_incomplete_artifact_ids):
            raise ValueError("validated and upstream-incomplete artifact IDs overlap")
        expected_outcome = deterministic_item_validation_outcome_v2(
            self.artifact_validation_results,
            self.upstream_incomplete_artifact_ids,
            self.candidate_inventory_failure_code,
        )
        if self.outcome is not expected_outcome:
            raise ValueError("deterministic item validation outcome is not exact")
        if self.candidate_inventory is None:
            if self.candidate_inventory_ref is not None:
                raise ValueError("candidate inventory ref requires nested inventory")
            if (
                self.artifact_validation_results
                and not expected_blocked
                and self.candidate_inventory_failure_code is None
            ):
                raise ValueError("complete candidate validations require candidate inventory")
        else:
            if self.candidate_inventory_failure_code is not None:
                raise ValueError("complete candidate inventory cannot carry failure code")
            expected_inventory_ref = candidate_package_inventory_ref(self.candidate_inventory)
            if self.candidate_inventory_ref != expected_inventory_ref:
                raise ValueError("candidate inventory ref must match nested inventory")
            if (
                self.candidate_inventory.attachment_reconstruction_result_ref
                != self.attachment_reconstruction_result_ref
                or self.candidate_inventory.artifact_validation_result_refs
                != self.artifact_validation_result_refs
            ):
                raise ValueError("candidate inventory does not match item validation sources")
        if self.accepted_artifact_refs:
            raise ValueError("accepted artifact refs remain empty before semantic review")
        expected_audit_refs = [
            self.attachment_reconstruction_result_ref,
            self.producer_task_view_ref,
            self.leakage_reference_set_ref,
            *self.artifact_validation_result_refs,
            *self.finding_refs,
        ]
        if self.candidate_inventory_ref is not None:
            expected_audit_refs.append(self.candidate_inventory_ref)
        _validate_audit(
            self.audit,
            tuple(sorted(expected_audit_refs, key=_ref_key)),
            "deterministic item validation",
        )
        return self


def deterministic_validation_finding_carried_sha256(
    finding: DeterministicValidationFindingV2,
) -> str:
    return _payload_sha256(
        {
            "artifact_build_result_ref": _ref_payload(finding.artifact_build_result_ref),
            "output_ref": _ref_payload(finding.output_ref),
            "facade_validation_result_ref": _ref_payload(finding.facade_validation_result_ref),
            "facade_finding_ref": _ref_payload(finding.facade_finding_ref),
            "inventory_member_ref": _maybe_ref_payload(finding.inventory_member_ref),
            "severity": finding.severity.value,
            "category": finding.category.value,
            "code": finding.code.value,
            "rule_ids": list(finding.rule_ids),
            "matched_fingerprint_ids": list(finding.matched_fingerprint_ids),
            "status": finding.status,
            "non_waivable": finding.non_waivable,
            "policy_version": finding.policy_version,
        }
    )


def deterministic_validation_finding_ref(
    finding: DeterministicValidationFindingV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="deterministic-validation-finding",
        object_id=finding.finding_id,
        object_version="v2",
        object_sha256=finding.finding_sha256,
    )


def validate_deterministic_validation_finding_identity(
    finding: DeterministicValidationFindingV2,
) -> None:
    _validate_identity(
        object_id=finding.finding_id,
        object_sha256=finding.finding_sha256,
        expected_prefix="deterministic-validation-finding",
        observed=deterministic_validation_finding_carried_sha256(finding),
    )


def artifact_deterministic_validation_result_carried_sha256(
    result: ArtifactDeterministicValidationResultV2,
) -> str:
    return _payload_sha256(
        {
            "attachment_reconstruction_result_ref": _ref_payload(result.attachment_reconstruction_result_ref),
            "artifact_build_result_ref": _ref_payload(result.artifact_build_result_ref),
            "build_spec_ref": _ref_payload(result.build_spec_ref),
            "output_ref": _ref_payload(result.output_ref),
            "output_sha256": result.output_sha256,
            "facade_validation_request_ref": _ref_payload(result.facade_validation_request_ref),
            "facade_validation_result_ref": _ref_payload(
                _object_ref_from_facade(attachment_validation_result_ref(result.facade_validation_result))
            ),
            "outcome": result.outcome.value,
            "finding_refs": [_ref_payload(ref) for ref in result.finding_refs],
            "inventory_member_refs": [
                _facade_ref_payload(attachment_inventory_member_ref(item))
                for item in result.inventory_members
            ],
            "policy_version": result.policy_version,
        }
    )


def artifact_deterministic_validation_result_ref(
    result: ArtifactDeterministicValidationResultV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="artifact-deterministic-validation-result",
        object_id=result.artifact_validation_result_id,
        object_version="v2",
        object_sha256=result.artifact_validation_result_sha256,
    )


def validate_artifact_deterministic_validation_result_identity(
    result: ArtifactDeterministicValidationResultV2,
) -> None:
    for finding in result.findings:
        validate_deterministic_validation_finding_identity(finding)
    _validate_identity(
        object_id=result.artifact_validation_result_id,
        object_sha256=result.artifact_validation_result_sha256,
        expected_prefix="artifact-deterministic-validation-result",
        observed=artifact_deterministic_validation_result_carried_sha256(result),
    )


def candidate_package_inventory_entry_carried_sha256(
    entry: CandidatePackageInventoryEntryV2,
) -> str:
    return _payload_sha256(
        {
            "inventory_member": entry.inventory_member.model_dump(
                mode="json",
                exclude_none=False,
            ),
            "inventory_member_ref": _ref_payload(entry.inventory_member_ref),
            "container_ref": _maybe_ref_payload(entry.container_ref),
            "artifact_build_result_ref": _ref_payload(entry.artifact_build_result_ref),
            "artifact_validation_result_ref": _ref_payload(entry.artifact_validation_result_ref),
            "output_ref": _ref_payload(entry.output_ref),
            "build_spec_ref": _ref_payload(entry.build_spec_ref),
            "derivation_root_refs": [_ref_payload(ref) for ref in entry.derivation_root_refs],
        }
    )


def candidate_package_inventory_entry_ref(
    entry: CandidatePackageInventoryEntryV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="candidate-package-inventory-entry",
        object_id=(f"candidate-package-inventory-entry://sha256/{entry.entry_sha256}"),
        object_version="v2",
        object_sha256=entry.entry_sha256,
    )


def validate_candidate_package_inventory_entry_identity(
    entry: CandidatePackageInventoryEntryV2,
) -> None:
    observed = candidate_package_inventory_entry_carried_sha256(entry)
    if observed != entry.entry_sha256:
        raise ValueError("candidate package inventory entry identity is stale")


def candidate_package_inventory_carried_sha256(
    inventory: CandidatePackageInventoryV2,
) -> str:
    return _payload_sha256(
        {
            "attachment_reconstruction_result_ref": _ref_payload(
                inventory.attachment_reconstruction_result_ref
            ),
            "artifact_validation_result_refs": [
                _ref_payload(ref) for ref in inventory.artifact_validation_result_refs
            ],
            "entry_refs": [
                _ref_payload(candidate_package_inventory_entry_ref(item)) for item in inventory.entries
            ],
            "exact_candidate_set_verified": (inventory.exact_candidate_set_verified),
            "final_package": inventory.final_package,
            "provenance_manifest_ref": None,
            "package_sha256": None,
            "input_state_only": None,
            "policy_version": inventory.policy_version,
        }
    )


def candidate_package_inventory_ref(
    inventory: CandidatePackageInventoryV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="candidate-package-inventory",
        object_id=inventory.candidate_inventory_id,
        object_version="v2",
        object_sha256=inventory.candidate_inventory_sha256,
    )


def validate_candidate_package_inventory_identity(
    inventory: CandidatePackageInventoryV2,
) -> None:
    for entry in inventory.entries:
        validate_candidate_package_inventory_entry_identity(entry)
    _validate_identity(
        object_id=inventory.candidate_inventory_id,
        object_sha256=inventory.candidate_inventory_sha256,
        expected_prefix="candidate-package-inventory",
        observed=candidate_package_inventory_carried_sha256(inventory),
    )


def deterministic_item_validation_result_carried_sha256(
    result: DeterministicItemValidationResultV2,
) -> str:
    return _payload_sha256(
        {
            "attachment_reconstruction_result_ref": _ref_payload(result.attachment_reconstruction_result_ref),
            "producer_task_view_ref": _ref_payload(result.producer_task_view_ref),
            "leakage_reference_set_ref": _ref_payload(result.leakage_reference_set_ref),
            "configured_pii_rules_sha256": (result.configured_pii_rules_sha256),
            "artifact_validation_result_refs": [
                _ref_payload(ref) for ref in result.artifact_validation_result_refs
            ],
            "finding_refs": [_ref_payload(ref) for ref in result.finding_refs],
            "candidate_inventory_ref": _maybe_ref_payload(result.candidate_inventory_ref),
            "candidate_inventory_failure_code": (
                result.candidate_inventory_failure_code.value
                if result.candidate_inventory_failure_code is not None
                else None
            ),
            "outcome": result.outcome.value,
            "validated_artifact_ids": list(result.validated_artifact_ids),
            "repair_required_artifact_ids": list(result.repair_required_artifact_ids),
            "rejected_artifact_ids": list(result.rejected_artifact_ids),
            "validation_blocked_artifact_ids": list(result.validation_blocked_artifact_ids),
            "upstream_incomplete_artifact_ids": list(result.upstream_incomplete_artifact_ids),
            "accepted_artifact_refs": [],
            "environment_spec_ref": None,
            "provenance_manifest_ref": None,
            "quality_report_ref": None,
            "package_sha256": None,
            "input_state_only": None,
            "policy_version": result.policy_version,
        }
    )


def deterministic_item_validation_result_ref(
    result: DeterministicItemValidationResultV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="deterministic-item-validation-result",
        object_id=result.deterministic_item_validation_result_id,
        object_version="v2",
        object_sha256=(result.deterministic_item_validation_result_sha256),
    )


def validate_deterministic_item_validation_result_identity(
    result: DeterministicItemValidationResultV2,
) -> None:
    for artifact_result in result.artifact_validation_results:
        validate_artifact_deterministic_validation_result_identity(artifact_result)
    if result.candidate_inventory is not None:
        validate_candidate_package_inventory_identity(result.candidate_inventory)
    _validate_identity(
        object_id=result.deterministic_item_validation_result_id,
        object_sha256=(result.deterministic_item_validation_result_sha256),
        expected_prefix="deterministic-item-validation-result",
        observed=deterministic_item_validation_result_carried_sha256(result),
    )


def deterministic_item_validation_outcome_v2(
    results: tuple[ArtifactDeterministicValidationResultV2, ...],
    upstream_incomplete_artifact_ids: tuple[str, ...],
    candidate_inventory_failure_code: (AttachmentValidationFailureCodeV2 | None) = None,
) -> DeterministicItemValidationOutcomeV2:
    outcomes = {item.outcome for item in results}
    if not results and not upstream_incomplete_artifact_ids:
        return DeterministicItemValidationOutcomeV2.NOT_REQUIRED
    if candidate_inventory_failure_code is not None:
        return DeterministicItemValidationOutcomeV2.BLOCKED
    if ArtifactDeterministicValidationOutcomeV2.BLOCKED in outcomes:
        return DeterministicItemValidationOutcomeV2.BLOCKED
    if ArtifactDeterministicValidationOutcomeV2.REJECTED in outcomes:
        return DeterministicItemValidationOutcomeV2.REJECTED
    if ArtifactDeterministicValidationOutcomeV2.REQUIRES_REPAIR in outcomes:
        return DeterministicItemValidationOutcomeV2.REQUIRES_REPAIR
    if upstream_incomplete_artifact_ids:
        return DeterministicItemValidationOutcomeV2.UPSTREAM_INCOMPLETE
    return DeterministicItemValidationOutcomeV2.PASSED


def _artifact_outcome(
    result: AttachmentValidationResultV2,
    findings: tuple[DeterministicValidationFindingV2, ...],
) -> ArtifactDeterministicValidationOutcomeV2:
    if result.status is AttachmentValidationStatusV2.BLOCKED:
        return ArtifactDeterministicValidationOutcomeV2.BLOCKED
    if result.status is AttachmentValidationStatusV2.PASSED:
        return ArtifactDeterministicValidationOutcomeV2.PASSED
    if any(item.non_waivable for item in findings):
        return ArtifactDeterministicValidationOutcomeV2.REJECTED
    return ArtifactDeterministicValidationOutcomeV2.REQUIRES_REPAIR


def _artifact_ids_for_outcome(
    results: tuple[ArtifactDeterministicValidationResultV2, ...],
    outcomes: set[ArtifactDeterministicValidationOutcomeV2],
) -> tuple[str, ...]:
    return tuple(
        sorted(item.facade_validation_result.artifact_id for item in results if item.outcome in outcomes)
    )


def _candidate_entry_key(
    entry: CandidatePackageInventoryEntryV2,
) -> tuple[str, str, str]:
    return (
        entry.container_ref.object_id if entry.container_ref is not None else "",
        entry.inventory_member.normalized_path,
        entry.inventory_member.member_type,
    )


def _validate_identity(
    *,
    object_id: str,
    object_sha256: str,
    expected_prefix: str,
    observed: str,
) -> None:
    if object_sha256 != observed or object_id != f"{expected_prefix}://sha256/{observed}":
        raise ValueError(f"{expected_prefix} identity is stale or invalid")


def _validate_audit(
    audit: ContractAudit,
    expected_refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != expected_refs:
        raise ValueError(f"{label} audit refs are not exact")
    if not any(
        item.component == "deterministic-validation" and item.version == "r5-08"
        for item in audit.governing_versions
    ):
        raise ValueError(f"{label} deterministic validation governing version is missing")


def _require_ref(
    ref: ObjectRef,
    expected_type: str,
    expected_version: str,
    field_name: str,
) -> None:
    if ref.object_type != expected_type or ref.object_version != expected_version:
        raise ValueError(f"{field_name} must reference {expected_type} {expected_version}")


def _require_ref_one_of(
    ref: ObjectRef,
    expected_types: tuple[str, ...],
    expected_version: str,
    field_name: str,
) -> None:
    if ref.object_type not in expected_types or ref.object_version != expected_version:
        raise ValueError(f"{field_name} must reference one of {expected_types} {expected_version}")


def _require_sorted_unique(
    label: str,
    values: tuple[object, ...],
) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")
    if values != tuple(sorted(values, key=repr)):
        raise ValueError(f"{label} must be sorted")


def _require_sorted_unique_refs(
    label: str,
    refs: tuple[ObjectRef, ...],
) -> None:
    _require_sorted_unique(label, tuple(_ref_key(ref) for ref in refs))


def _object_ref_from_facade(ref: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=False)


def _facade_ref_payload(ref: FacadeObjectRef) -> dict[str, object]:
    return dict(ref.model_dump(mode="json", exclude_none=False))


def _maybe_ref_payload(
    ref: ObjectRef | None,
) -> dict[str, object] | None:
    return None if ref is None else _ref_payload(ref)


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
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
