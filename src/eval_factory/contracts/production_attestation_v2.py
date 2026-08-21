from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2

PRODUCTION_READINESS_ATTESTATION_POLICY_VERSION: Literal["production-readiness-attestation/r8-08-v1"] = (
    "production-readiness-attestation/r8-08-v1"
)


class ProductionAttestationEvidenceClassV2(StrEnum):
    PRODUCTION_VERIFIED = "PRODUCTION_VERIFIED"
    MECHANISM_VALIDATION_ONLY = "MECHANISM_VALIDATION_ONLY"
    REPOSITORY_PENDING_ONLY = "REPOSITORY_PENDING_ONLY"


class ProductionReadinessGateV2(StrEnum):
    SC_010_LABEL_QUALITY = "SC_010_LABEL_QUALITY"
    SC_011_REAL_TRACE_STABILITY = "SC_011_REAL_TRACE_STABILITY"
    SC_012_RESOURCE_POLICY = "SC_012_RESOURCE_POLICY"
    SC_013_ORGANIZATION_APPROVALS = "SC_013_ORGANIZATION_APPROVALS"
    SC_014_ZERO_OPEN_GATES = "SC_014_ZERO_OPEN_GATES"


class ProductionReadinessGateOutcomeV2(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    PENDING = "PENDING"


class ProductionReadinessAttestationOutcomeV2(StrEnum):
    ISSUED = "ISSUED"
    ATTESTATION_PENDING = "ATTESTATION_PENDING"
    REJECTED = "REJECTED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


class ProductionReadinessAttestationStateV2(StrEnum):
    ACTIVE = "ACTIVE"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


class ProductionReadinessInvalidationTriggerV2(StrEnum):
    SYSTEM_VERSION_CHANGED = "SYSTEM_VERSION_CHANGED"
    BASE_CONTRACT_MANIFEST_CHANGED = "BASE_CONTRACT_MANIFEST_CHANGED"
    OVERLAY_CONTRACT_MANIFEST_CHANGED = "OVERLAY_CONTRACT_MANIFEST_CHANGED"
    SCHEMA_SET_CHANGED = "SCHEMA_SET_CHANGED"
    POLICY_SET_CHANGED = "POLICY_SET_CHANGED"
    EVIDENCE_SUPERSEDED = "EVIDENCE_SUPERSEDED"
    ISSUER_REGISTRY_CHANGED = "ISSUER_REGISTRY_CHANGED"
    ISSUER_REVOKED_OR_EXPIRED = "ISSUER_REVOKED_OR_EXPIRED"
    ATTESTATION_EXPIRED = "ATTESTATION_EXPIRED"
    EXPLICIT_INCIDENT_REVOCATION = "EXPLICIT_INCIDENT_REVOCATION"
    EXPLICIT_RELEASE_REVOCATION = "EXPLICIT_RELEASE_REVOCATION"


class ProductionReadinessAttestationReasonCodeV2(StrEnum):
    NONE = "NONE"
    SC_010_PENDING = "SC_010_PENDING"
    SC_011_PENDING = "SC_011_PENDING"
    SC_012_PENDING = "SC_012_PENDING"
    SC_013_PENDING = "SC_013_PENDING"
    SC_014_UNVERIFIED = "SC_014_UNVERIFIED"
    GATE_FAILED = "GATE_FAILED"
    LIFECYCLE_APPROVAL_MISSING = "LIFECYCLE_APPROVAL_MISSING"
    INCIDENT_ROUTE_MISSING = "INCIDENT_ROUTE_MISSING"
    ISSUER_AUTHORITY_MISSING = "ISSUER_AUTHORITY_MISSING"
    ISSUER_AUTHORITY_EXPIRED = "ISSUER_AUTHORITY_EXPIRED"
    ISSUER_SCOPE_MISMATCH = "ISSUER_SCOPE_MISMATCH"
    VERSION_SET_DRIFT = "VERSION_SET_DRIFT"
    EVIDENCE_EXPIRED = "EVIDENCE_EXPIRED"
    MECHANISM_ONLY = "MECHANISM_ONLY"
    TRUSTED_REJECTION = "TRUSTED_REJECTION"
    ATTESTATION_EXPIRED = "ATTESTATION_EXPIRED"
    ATTESTATION_INVALIDATED = "ATTESTATION_INVALIDATED"


PRODUCTION_ATTESTATION_GATE_ORDER = tuple(ProductionReadinessGateV2)
PRODUCTION_ATTESTATION_INVALIDATION_TRIGGER_ORDER = tuple(ProductionReadinessInvalidationTriggerV2)


class ProductionReadinessInvalidationPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-invalidation-policy/v2"] = (
        "eval-factory/production-readiness-invalidation-policy/v2"
    )
    policy_id: Identifier
    required_triggers: tuple[ProductionReadinessInvalidationTriggerV2, ...] = (
        PRODUCTION_ATTESTATION_INVALIDATION_TRIGGER_ORDER
    )
    max_projection_revisions: int = Field(ge=2, le=1_000_000)
    max_invalidation_refs: int = Field(ge=1, le=10_000)
    policy_version: Literal["production-readiness-attestation/r8-08-v1"] = (
        PRODUCTION_READINESS_ATTESTATION_POLICY_VERSION
    )
    policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.required_triggers != PRODUCTION_ATTESTATION_INVALIDATION_TRIGGER_ORDER:
            raise ValueError("invalidation trigger inventory is not canonical")
        _require_audit(self.audit, (), "production invalidation policy")
        _validate_identity(
            self.policy_id,
            self.policy_sha256,
            "production-readiness-invalidation-policy",
            production_readiness_invalidation_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        max_projection_revisions: int,
        max_invalidation_refs: int,
        audit: ContractAudit,
    ) -> ProductionReadinessInvalidationPolicyV2:
        value = cls(
            policy_id="production-readiness-invalidation-policy://pending",
            max_projection_revisions=max_projection_revisions,
            max_invalidation_refs=max_invalidation_refs,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, ()),
        )
        return _finalize(
            value,
            "policy_id",
            "policy_sha256",
            "production-readiness-invalidation-policy",
            production_readiness_invalidation_policy_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return production_readiness_invalidation_policy_v2_ref(self)


class ProductionReadinessVersionSetV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-version-set/v2"] = (
        "eval-factory/production-readiness-version-set/v2"
    )
    version_set_id: Identifier
    system_version: str = Field(min_length=1, max_length=128)
    base_contract_manifest_ref: ObjectRef
    overlay_contract_manifest_ref: ObjectRef
    schema_manifest_refs: tuple[ObjectRef, ...] = Field(min_length=2, max_length=128)
    policy_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=1_000)
    release_profile_decision_ref: ObjectRef
    resource_policy_approval_ref: ObjectRef | None = None
    lifecycle_approval_ref: ObjectRef | None = None
    incident_route_ref: ObjectRef | None = None
    version_set_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_version_set(self) -> Self:
        _require_ref(
            self.base_contract_manifest_ref,
            "contract-manifest",
            "v1",
            "base_contract_manifest_ref",
        )
        _require_ref(
            self.overlay_contract_manifest_ref,
            "contract-manifest",
            "v2",
            "overlay_contract_manifest_ref",
        )
        _require_sorted_unique_refs(
            self.schema_manifest_refs,
            "schema manifest refs",
        )
        _require_sorted_unique_refs(self.policy_refs, "policy refs")
        _require_ref(
            self.release_profile_decision_ref,
            "release-profile-decision",
            "v1",
            "release_profile_decision_ref",
        )
        production_refs = (
            self.resource_policy_approval_ref,
            self.lifecycle_approval_ref,
            self.incident_route_ref,
        )
        if any(item is None for item in production_refs) and any(
            item is not None for item in production_refs
        ):
            raise ValueError("production resource/lifecycle/incident authority must be all-or-none")
        if self.resource_policy_approval_ref is not None:
            _require_ref(
                self.resource_policy_approval_ref,
                "production-resource-policy-approval",
                "v1",
                "resource_policy_approval_ref",
            )
        if self.lifecycle_approval_ref is not None:
            _require_ref(
                self.lifecycle_approval_ref,
                "data-lifecycle-approval",
                "v1",
                "lifecycle_approval_ref",
            )
        if self.incident_route_ref is not None:
            _require_ref(
                self.incident_route_ref,
                "incident-route",
                "v1",
                "incident_route_ref",
            )
        refs = self.refs
        _require_audit(self.audit, refs, "production version set")
        _validate_identity(
            self.version_set_id,
            self.version_set_sha256,
            "production-readiness-version-set",
            production_readiness_version_set_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        system_version: str,
        base_contract_manifest_ref: ObjectRef,
        overlay_contract_manifest_ref: ObjectRef,
        schema_manifest_refs: tuple[ObjectRef, ...],
        policy_refs: tuple[ObjectRef, ...],
        release_profile_decision_ref: ObjectRef,
        resource_policy_approval_ref: ObjectRef | None,
        lifecycle_approval_ref: ObjectRef | None,
        incident_route_ref: ObjectRef | None,
        audit: ContractAudit,
    ) -> ProductionReadinessVersionSetV2:
        schemas = _sorted_refs(schema_manifest_refs)
        policies = _sorted_refs(policy_refs)
        refs = (
            base_contract_manifest_ref,
            overlay_contract_manifest_ref,
            *schemas,
            *policies,
            release_profile_decision_ref,
            *((resource_policy_approval_ref,) if resource_policy_approval_ref is not None else ()),
            *((lifecycle_approval_ref,) if lifecycle_approval_ref is not None else ()),
            *((incident_route_ref,) if incident_route_ref is not None else ()),
        )
        value = cls(
            version_set_id="production-readiness-version-set://pending",
            system_version=system_version,
            base_contract_manifest_ref=base_contract_manifest_ref,
            overlay_contract_manifest_ref=overlay_contract_manifest_ref,
            schema_manifest_refs=schemas,
            policy_refs=policies,
            release_profile_decision_ref=release_profile_decision_ref,
            resource_policy_approval_ref=resource_policy_approval_ref,
            lifecycle_approval_ref=lifecycle_approval_ref,
            incident_route_ref=incident_route_ref,
            version_set_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "version_set_id",
            "version_set_sha256",
            "production-readiness-version-set",
            production_readiness_version_set_v2_carried_sha256(value),
        )

    @property
    def refs(self) -> tuple[ObjectRef, ...]:
        return (
            self.base_contract_manifest_ref,
            self.overlay_contract_manifest_ref,
            *self.schema_manifest_refs,
            *self.policy_refs,
            self.release_profile_decision_ref,
            *((self.resource_policy_approval_ref,) if self.resource_policy_approval_ref is not None else ()),
            *((self.lifecycle_approval_ref,) if self.lifecycle_approval_ref is not None else ()),
            *((self.incident_route_ref,) if self.incident_route_ref is not None else ()),
        )

    def to_ref(self) -> ObjectRef:
        return production_readiness_version_set_v2_ref(self)


class ProductionReadinessAttestationPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-attestation-policy/v2"] = (
        "eval-factory/production-readiness-attestation-policy/v2"
    )
    policy_id: Identifier
    expected_system_version: str = Field(min_length=1, max_length=128)
    base_contract_manifest_ref: ObjectRef
    overlay_contract_manifest_ref: ObjectRef
    required_schema_manifest_refs: tuple[ObjectRef, ...] = Field(
        min_length=2,
        max_length=128,
    )
    required_policy_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=1_000)
    release_profile_decision_ref: ObjectRef
    repository_pending_evidence_ref: ObjectRef
    invalidation_policy: ProductionReadinessInvalidationPolicyV2
    trusted_issuer_registry_ref: ObjectRef | None = None
    evidence_class: ProductionAttestationEvidenceClassV2
    required_gates: tuple[ProductionReadinessGateV2, ...] = PRODUCTION_ATTESTATION_GATE_ORDER
    minimum_issuer_count: int = Field(ge=2, le=100)
    maximum_validity_seconds: int = Field(ge=1, le=10 * 365 * 24 * 60 * 60)
    max_evidence_refs: int = Field(ge=5, le=100_000)
    max_private_bytes: int = Field(ge=2, le=10**12)
    max_report_bytes: int = Field(ge=2, le=100_000_000)
    policy_version: Literal["production-readiness-attestation/r8-08-v1"] = (
        PRODUCTION_READINESS_ATTESTATION_POLICY_VERSION
    )
    policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.required_gates != PRODUCTION_ATTESTATION_GATE_ORDER:
            raise ValueError("production attestation gates are not canonical")
        _require_ref(
            self.base_contract_manifest_ref,
            "contract-manifest",
            "v1",
            "base_contract_manifest_ref",
        )
        _require_ref(
            self.overlay_contract_manifest_ref,
            "contract-manifest",
            "v2",
            "overlay_contract_manifest_ref",
        )
        _require_sorted_unique_refs(
            self.required_schema_manifest_refs,
            "required schema manifest refs",
        )
        _require_sorted_unique_refs(self.required_policy_refs, "required policy refs")
        _require_ref(
            self.release_profile_decision_ref,
            "release-profile-decision",
            "v1",
            "release_profile_decision_ref",
        )
        _require_ref(
            self.repository_pending_evidence_ref,
            "production-attestation-repository-evidence",
            "json/v1",
            "repository_pending_evidence_ref",
        )
        validate_production_readiness_invalidation_policy_v2_identity(self.invalidation_policy)
        if self.evidence_class is ProductionAttestationEvidenceClassV2.PRODUCTION_VERIFIED:
            if self.trusted_issuer_registry_ref is None:
                raise ValueError("production attestation policy requires issuer registry")
            _require_ref(
                self.trusted_issuer_registry_ref,
                "attestation-issuer-registry",
                "private-v1",
                "trusted_issuer_registry_ref",
            )
        elif self.trusted_issuer_registry_ref is not None:
            raise ValueError("non-production policy cannot bind issuer registry")
        refs = self.refs
        _require_audit(self.audit, refs, "production attestation policy")
        _validate_identity(
            self.policy_id,
            self.policy_sha256,
            "production-readiness-attestation-policy",
            production_readiness_attestation_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        expected_system_version: str,
        base_contract_manifest_ref: ObjectRef,
        overlay_contract_manifest_ref: ObjectRef,
        required_schema_manifest_refs: tuple[ObjectRef, ...],
        required_policy_refs: tuple[ObjectRef, ...],
        release_profile_decision_ref: ObjectRef,
        repository_pending_evidence_ref: ObjectRef,
        invalidation_policy: ProductionReadinessInvalidationPolicyV2,
        trusted_issuer_registry_ref: ObjectRef | None,
        evidence_class: ProductionAttestationEvidenceClassV2,
        minimum_issuer_count: int,
        maximum_validity_seconds: int,
        max_evidence_refs: int,
        max_private_bytes: int,
        max_report_bytes: int,
        audit: ContractAudit,
    ) -> ProductionReadinessAttestationPolicyV2:
        schemas = _sorted_refs(required_schema_manifest_refs)
        policies = _sorted_refs(required_policy_refs)
        refs = (
            base_contract_manifest_ref,
            overlay_contract_manifest_ref,
            *schemas,
            *policies,
            release_profile_decision_ref,
            repository_pending_evidence_ref,
            invalidation_policy.to_ref(),
            *((trusted_issuer_registry_ref,) if trusted_issuer_registry_ref is not None else ()),
        )
        value = cls(
            policy_id="production-readiness-attestation-policy://pending",
            expected_system_version=expected_system_version,
            base_contract_manifest_ref=base_contract_manifest_ref,
            overlay_contract_manifest_ref=overlay_contract_manifest_ref,
            required_schema_manifest_refs=schemas,
            required_policy_refs=policies,
            release_profile_decision_ref=release_profile_decision_ref,
            repository_pending_evidence_ref=repository_pending_evidence_ref,
            invalidation_policy=invalidation_policy,
            trusted_issuer_registry_ref=trusted_issuer_registry_ref,
            evidence_class=evidence_class,
            minimum_issuer_count=minimum_issuer_count,
            maximum_validity_seconds=maximum_validity_seconds,
            max_evidence_refs=max_evidence_refs,
            max_private_bytes=max_private_bytes,
            max_report_bytes=max_report_bytes,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "policy_id",
            "policy_sha256",
            "production-readiness-attestation-policy",
            production_readiness_attestation_policy_v2_carried_sha256(value),
        )

    @property
    def refs(self) -> tuple[ObjectRef, ...]:
        return (
            self.base_contract_manifest_ref,
            self.overlay_contract_manifest_ref,
            *self.required_schema_manifest_refs,
            *self.required_policy_refs,
            self.release_profile_decision_ref,
            self.repository_pending_evidence_ref,
            self.invalidation_policy.to_ref(),
            *((self.trusted_issuer_registry_ref,) if self.trusted_issuer_registry_ref is not None else ()),
        )

    def to_ref(self) -> ObjectRef:
        return production_readiness_attestation_policy_v2_ref(self)

    @property
    def invalidation_policy_ref(self) -> ObjectRef:
        return self.invalidation_policy.to_ref()


class ProductionReadinessGateAssessmentV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-gate-assessment/v2"] = (
        "eval-factory/production-readiness-gate-assessment/v2"
    )
    assessment_id: Identifier
    gate: ProductionReadinessGateV2
    outcome: ProductionReadinessGateOutcomeV2
    reason_codes: tuple[ProductionReadinessAttestationReasonCodeV2, ...] = Field(min_length=1)
    evidence_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=10_000)
    valid_until: datetime | None = None
    assessment_sha256: Sha256

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(
        cls,
        value: tuple[ProductionReadinessAttestationReasonCodeV2, ...],
    ) -> tuple[ProductionReadinessAttestationReasonCodeV2, ...]:
        expected = tuple(sorted(set(value), key=lambda item: item.value))
        if value != expected:
            raise ValueError("gate reason codes must be sorted and unique")
        if ProductionReadinessAttestationReasonCodeV2.NONE in value and value != (
            ProductionReadinessAttestationReasonCodeV2.NONE,
        ):
            raise ValueError("NONE cannot accompany another gate reason")
        return value

    @model_validator(mode="after")
    def validate_assessment(self) -> Self:
        _require_sorted_unique_refs(self.evidence_refs, "gate evidence refs")
        if self.outcome is ProductionReadinessGateOutcomeV2.PASSED:
            if (
                self.reason_codes != (ProductionReadinessAttestationReasonCodeV2.NONE,)
                or self.valid_until is None
            ):
                raise ValueError("passed gate requires NONE and validity")
            _require_aware(self.valid_until, "gate valid_until")
        elif (
            ProductionReadinessAttestationReasonCodeV2.NONE in self.reason_codes
            or self.valid_until is not None
        ):
            raise ValueError("non-passed gate cannot carry pass authority")
        _validate_identity(
            self.assessment_id,
            self.assessment_sha256,
            "production-readiness-gate-assessment",
            production_readiness_gate_assessment_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        gate: ProductionReadinessGateV2,
        outcome: ProductionReadinessGateOutcomeV2,
        reason_codes: tuple[ProductionReadinessAttestationReasonCodeV2 | str, ...],
        evidence_refs: tuple[ObjectRef, ...],
        valid_until: datetime | None,
    ) -> ProductionReadinessGateAssessmentV2:
        reasons = tuple(
            sorted(
                {
                    value
                    if isinstance(value, ProductionReadinessAttestationReasonCodeV2)
                    else ProductionReadinessAttestationReasonCodeV2(value)
                    for value in reason_codes
                },
                key=lambda value: value.value,
            )
        )
        value = cls(
            assessment_id="production-readiness-gate-assessment://pending",
            gate=gate,
            outcome=outcome,
            reason_codes=reasons,
            evidence_refs=_sorted_refs(evidence_refs),
            valid_until=valid_until,
            assessment_sha256="0" * 64,
        )
        return _finalize(
            value,
            "assessment_id",
            "assessment_sha256",
            "production-readiness-gate-assessment",
            production_readiness_gate_assessment_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return production_readiness_gate_assessment_v2_ref(self)


class ProductionReadinessAttestationPrerequisiteV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-attestation-prerequisite/v2"] = (
        "eval-factory/production-readiness-attestation-prerequisite/v2"
    )
    prerequisite_id: Identifier
    gate_assessments: tuple[ProductionReadinessGateAssessmentV2, ...] = Field(
        min_length=5,
        max_length=5,
    )
    passed_gate_count: int = Field(ge=0, le=5)
    failed_gate_count: int = Field(ge=0, le=5)
    pending_gate_count: int = Field(ge=0, le=5)
    outcome: ProductionReadinessAttestationOutcomeV2
    reason_codes: tuple[ProductionReadinessAttestationReasonCodeV2, ...] = Field(min_length=1)
    valid_until: datetime | None = None
    prerequisite_sha256: Sha256
    audit: ContractAudit

    @field_validator("gate_assessments")
    @classmethod
    def validate_gate_order(
        cls,
        value: tuple[ProductionReadinessGateAssessmentV2, ...],
    ) -> tuple[ProductionReadinessGateAssessmentV2, ...]:
        if tuple(item.gate for item in value) != PRODUCTION_ATTESTATION_GATE_ORDER:
            raise ValueError("gate assessment order is not canonical")
        return value

    @model_validator(mode="after")
    def validate_prerequisite(self) -> Self:
        counts = _gate_counts(self.gate_assessments)
        if (
            self.passed_gate_count,
            self.failed_gate_count,
            self.pending_gate_count,
        ) != counts:
            raise ValueError("gate counts are not derived")
        outcome, reasons, valid_until = _prerequisite_outcome(self.gate_assessments)
        if self.outcome is not outcome or self.reason_codes != reasons or self.valid_until != valid_until:
            raise ValueError("prerequisite outcome differs from gates")
        refs = tuple(item.to_ref() for item in self.gate_assessments)
        _require_audit(self.audit, refs, "production attestation prerequisite")
        _validate_identity(
            self.prerequisite_id,
            self.prerequisite_sha256,
            "production-readiness-attestation-prerequisite",
            production_readiness_attestation_prerequisite_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        gate_assessments: tuple[ProductionReadinessGateAssessmentV2, ...],
        audit: ContractAudit,
    ) -> ProductionReadinessAttestationPrerequisiteV2:
        ordered = tuple(
            sorted(
                gate_assessments,
                key=lambda item: PRODUCTION_ATTESTATION_GATE_ORDER.index(item.gate),
            )
        )
        counts = _gate_counts(ordered)
        outcome, reasons, valid_until = _prerequisite_outcome(ordered)
        refs = tuple(item.to_ref() for item in ordered)
        value = cls(
            prerequisite_id="production-readiness-attestation-prerequisite://pending",
            gate_assessments=ordered,
            passed_gate_count=counts[0],
            failed_gate_count=counts[1],
            pending_gate_count=counts[2],
            outcome=outcome,
            reason_codes=reasons,
            valid_until=valid_until,
            prerequisite_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "prerequisite_id",
            "prerequisite_sha256",
            "production-readiness-attestation-prerequisite",
            production_readiness_attestation_prerequisite_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return production_readiness_attestation_prerequisite_v2_ref(self)


class ProductionReadinessInvalidationRecordV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-invalidation-record/v2"] = (
        "eval-factory/production-readiness-invalidation-record/v2"
    )
    record_id: Identifier
    prior_projection_ref: ObjectRef
    prior_version_set_ref: ObjectRef
    observed_version_set_ref: ObjectRef
    triggers: tuple[ProductionReadinessInvalidationTriggerV2, ...] = Field(min_length=1)
    explicit_revocation_ref: ObjectRef | None = None
    observed_at: datetime
    record_sha256: Sha256
    audit: ContractAudit

    @field_validator("triggers")
    @classmethod
    def validate_triggers(
        cls,
        value: tuple[ProductionReadinessInvalidationTriggerV2, ...],
    ) -> tuple[ProductionReadinessInvalidationTriggerV2, ...]:
        expected = tuple(
            sorted(
                set(value),
                key=lambda item: PRODUCTION_ATTESTATION_INVALIDATION_TRIGGER_ORDER.index(item),
            )
        )
        if value != expected:
            raise ValueError("invalidation triggers must be canonical and unique")
        return value

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        _require_ref(
            self.prior_projection_ref,
            "production-readiness-attestation-projection",
            "v2",
            "prior_projection_ref",
        )
        _require_ref(
            self.prior_version_set_ref,
            "production-readiness-version-set",
            "v2",
            "prior_version_set_ref",
        )
        _require_ref(
            self.observed_version_set_ref,
            "production-readiness-version-set",
            "v2",
            "observed_version_set_ref",
        )
        _require_aware(self.observed_at, "invalidation observed_at")
        explicit = {
            ProductionReadinessInvalidationTriggerV2.EXPLICIT_INCIDENT_REVOCATION,
            ProductionReadinessInvalidationTriggerV2.EXPLICIT_RELEASE_REVOCATION,
        }
        explicit_triggers = set(self.triggers).intersection(explicit)
        if len(explicit_triggers) > 1:
            raise ValueError("one invalidation record cannot carry two explicit revocations")
        if bool(explicit_triggers) != (self.explicit_revocation_ref is not None):
            raise ValueError("explicit invalidation trigger/ref closure is invalid")
        refs = (
            self.prior_projection_ref,
            self.prior_version_set_ref,
            self.observed_version_set_ref,
            *((self.explicit_revocation_ref,) if self.explicit_revocation_ref is not None else ()),
        )
        _require_audit(self.audit, refs, "production invalidation record")
        _validate_identity(
            self.record_id,
            self.record_sha256,
            "production-readiness-invalidation-record",
            production_readiness_invalidation_record_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        prior_projection_ref: ObjectRef,
        prior_version_set_ref: ObjectRef,
        observed_version_set_ref: ObjectRef,
        triggers: tuple[ProductionReadinessInvalidationTriggerV2, ...],
        explicit_revocation_ref: ObjectRef | None,
        observed_at: datetime,
        audit: ContractAudit,
    ) -> ProductionReadinessInvalidationRecordV2:
        ordered = tuple(
            sorted(
                set(triggers),
                key=lambda item: PRODUCTION_ATTESTATION_INVALIDATION_TRIGGER_ORDER.index(item),
            )
        )
        refs = (
            prior_projection_ref,
            prior_version_set_ref,
            observed_version_set_ref,
            *((explicit_revocation_ref,) if explicit_revocation_ref is not None else ()),
        )
        value = cls(
            record_id="production-readiness-invalidation-record://pending",
            prior_projection_ref=prior_projection_ref,
            prior_version_set_ref=prior_version_set_ref,
            observed_version_set_ref=observed_version_set_ref,
            triggers=ordered,
            explicit_revocation_ref=explicit_revocation_ref,
            observed_at=observed_at,
            record_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "record_id",
            "record_sha256",
            "production-readiness-invalidation-record",
            production_readiness_invalidation_record_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return production_readiness_invalidation_record_v2_ref(self)


class ProductionReadinessAttestationProjectionV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-attestation-projection/v2"] = (
        "eval-factory/production-readiness-attestation-projection/v2"
    )
    projection_id: Identifier
    attestation_series_id: Identifier
    attestation_version: int = Field(ge=1, le=1_000_000)
    projection_revision: int = Field(ge=1, le=1_000_000)
    previous_projection_ref: ObjectRef | None = None
    frozen_attestation_ref: ObjectRef
    version_set_ref: ObjectRef
    prerequisite_ref: ObjectRef
    issuer_registry_ref: ObjectRef
    issuer_count: int = Field(ge=2, le=100)
    valid_from: datetime
    valid_until: datetime
    state: ProductionReadinessAttestationStateV2
    invalidation_record_ref: ObjectRef | None = None
    projection_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_projection(self) -> Self:
        _require_ref(
            self.frozen_attestation_ref,
            "production-readiness-attestation",
            "v1",
            "frozen_attestation_ref",
        )
        _require_ref(
            self.version_set_ref,
            "production-readiness-version-set",
            "v2",
            "version_set_ref",
        )
        _require_ref(
            self.prerequisite_ref,
            "production-readiness-attestation-prerequisite",
            "v2",
            "prerequisite_ref",
        )
        _require_ref(
            self.issuer_registry_ref,
            "attestation-issuer-registry",
            "private-v1",
            "issuer_registry_ref",
        )
        _require_aware(self.valid_from, "projection valid_from")
        _require_aware(self.valid_until, "projection valid_until")
        if self.valid_until <= self.valid_from:
            raise ValueError("attestation projection validity is invalid")
        if self.projection_revision == 1:
            if (
                self.previous_projection_ref is not None
                or self.invalidation_record_ref is not None
                or self.state is not ProductionReadinessAttestationStateV2.ACTIVE
            ):
                raise ValueError("initial attestation projection must be active")
        else:
            if (
                self.previous_projection_ref is None
                or self.invalidation_record_ref is None
                or self.state is ProductionReadinessAttestationStateV2.ACTIVE
            ):
                raise ValueError("successor attestation projection cannot reactivate")
            _require_ref(
                self.previous_projection_ref,
                "production-readiness-attestation-projection",
                "v2",
                "previous_projection_ref",
            )
            _require_ref(
                self.invalidation_record_ref,
                "production-readiness-invalidation-record",
                "v2",
                "invalidation_record_ref",
            )
        refs = self.refs
        _require_audit(self.audit, refs, "production attestation projection")
        _validate_identity(
            self.projection_id,
            self.projection_sha256,
            "production-readiness-attestation-projection",
            production_readiness_attestation_projection_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create_active(
        cls,
        *,
        attestation_series_id: str,
        attestation_version: int,
        frozen_attestation_ref: ObjectRef,
        version_set_ref: ObjectRef,
        prerequisite_ref: ObjectRef,
        issuer_registry_ref: ObjectRef,
        issuer_count: int,
        valid_from: datetime,
        valid_until: datetime,
        audit: ContractAudit,
    ) -> ProductionReadinessAttestationProjectionV2:
        refs = (
            frozen_attestation_ref,
            version_set_ref,
            prerequisite_ref,
            issuer_registry_ref,
        )
        value = cls(
            projection_id="production-readiness-attestation-projection://pending",
            attestation_series_id=attestation_series_id,
            attestation_version=attestation_version,
            projection_revision=1,
            previous_projection_ref=None,
            frozen_attestation_ref=frozen_attestation_ref,
            version_set_ref=version_set_ref,
            prerequisite_ref=prerequisite_ref,
            issuer_registry_ref=issuer_registry_ref,
            issuer_count=issuer_count,
            valid_from=valid_from,
            valid_until=valid_until,
            state=ProductionReadinessAttestationStateV2.ACTIVE,
            invalidation_record_ref=None,
            projection_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "projection_id",
            "projection_sha256",
            "production-readiness-attestation-projection",
            production_readiness_attestation_projection_v2_carried_sha256(value),
        )

    @classmethod
    def create_successor(
        cls,
        *,
        prior: ProductionReadinessAttestationProjectionV2,
        state: ProductionReadinessAttestationStateV2,
        invalidation_record: ProductionReadinessInvalidationRecordV2,
        audit: ContractAudit,
    ) -> ProductionReadinessAttestationProjectionV2:
        if prior.state is not ProductionReadinessAttestationStateV2.ACTIVE:
            raise ValueError("invalidated attestation cannot reactivate")
        if state is ProductionReadinessAttestationStateV2.ACTIVE:
            raise ValueError("successor projection cannot reactivate")
        if invalidation_record.prior_projection_ref != prior.to_ref():
            raise ValueError("invalidation record differs from exact prior projection")
        refs = (
            prior.to_ref(),
            prior.frozen_attestation_ref,
            prior.version_set_ref,
            prior.prerequisite_ref,
            prior.issuer_registry_ref,
            invalidation_record.to_ref(),
        )
        value = cls(
            projection_id="production-readiness-attestation-projection://pending",
            attestation_series_id=prior.attestation_series_id,
            attestation_version=prior.attestation_version,
            projection_revision=prior.projection_revision + 1,
            previous_projection_ref=prior.to_ref(),
            frozen_attestation_ref=prior.frozen_attestation_ref,
            version_set_ref=prior.version_set_ref,
            prerequisite_ref=prior.prerequisite_ref,
            issuer_registry_ref=prior.issuer_registry_ref,
            issuer_count=prior.issuer_count,
            valid_from=prior.valid_from,
            valid_until=prior.valid_until,
            state=state,
            invalidation_record_ref=invalidation_record.to_ref(),
            projection_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "projection_id",
            "projection_sha256",
            "production-readiness-attestation-projection",
            production_readiness_attestation_projection_v2_carried_sha256(value),
        )

    @property
    def refs(self) -> tuple[ObjectRef, ...]:
        return (
            *((self.previous_projection_ref,) if self.previous_projection_ref is not None else ()),
            self.frozen_attestation_ref,
            self.version_set_ref,
            self.prerequisite_ref,
            self.issuer_registry_ref,
            *((self.invalidation_record_ref,) if self.invalidation_record_ref is not None else ()),
        )

    def to_ref(self) -> ObjectRef:
        return production_readiness_attestation_projection_v2_ref(self)


class ProductionReadinessAttestationResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-attestation-result/v2"] = (
        "eval-factory/production-readiness-attestation-result/v2"
    )
    result_id: Identifier
    policy_ref: ObjectRef
    version_set: ProductionReadinessVersionSetV2
    prerequisite: ProductionReadinessAttestationPrerequisiteV2
    evidence_class: ProductionAttestationEvidenceClassV2
    attestation_series_id: Identifier
    attestation_version: int = Field(ge=1, le=1_000_000)
    previous_result_ref: ObjectRef | None = None
    projection: ProductionReadinessAttestationProjectionV2 | None = None
    invalidation_record: ProductionReadinessInvalidationRecordV2 | None = None
    private_request_closure_sha256: Sha256 | None = None
    private_issuer_closure_sha256: Sha256 | None = None
    private_attestation_closure_sha256: Sha256 | None = None
    outcome: ProductionReadinessAttestationOutcomeV2
    reason_codes: tuple[ProductionReadinessAttestationReasonCodeV2, ...] = Field(min_length=1)
    passed_gate_count: int = Field(ge=0, le=5)
    failed_gate_count: int = Field(ge=0, le=5)
    pending_gate_count: int = Field(ge=0, le=5)
    valid_until: datetime | None = None
    satisfies_sc_010: bool
    satisfies_sc_011: bool
    satisfies_sc_012: bool
    satisfies_sc_013: bool
    satisfies_sc_014: bool
    active_attestation: bool
    satisfies_sc_015: Literal[False] = False
    authorizes_production_release: Literal[False] = False
    policy_version: Literal["production-readiness-attestation/r8-08-v1"] = (
        PRODUCTION_READINESS_ATTESTATION_POLICY_VERSION
    )
    result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.policy_ref,
            "production-readiness-attestation-policy",
            "v2",
            "policy_ref",
        )
        if self.attestation_version == 1:
            if self.previous_result_ref is not None:
                raise ValueError("attestation version 1 cannot have prior result")
        else:
            if self.previous_result_ref is None:
                raise ValueError("later attestation version requires prior result")
            _require_ref(
                self.previous_result_ref,
                "production-readiness-attestation-result",
                "v2",
                "previous_result_ref",
            )
        if self.projection is None:
            if self.invalidation_record is not None:
                raise ValueError("invalidation record requires an attestation projection")
        else:
            projection_authority = (
                self.projection.attestation_series_id,
                self.projection.attestation_version,
                self.projection.version_set_ref,
                self.projection.prerequisite_ref,
            )
            expected_projection_authority = (
                self.attestation_series_id,
                self.attestation_version,
                self.version_set.to_ref(),
                self.prerequisite.to_ref(),
            )
            if projection_authority != expected_projection_authority:
                raise ValueError("attestation projection authority differs from result")
            if self.invalidation_record is not None and (
                self.projection.invalidation_record_ref != self.invalidation_record.to_ref()
                or self.projection.previous_projection_ref != self.invalidation_record.prior_projection_ref
            ):
                raise ValueError("invalidation authority differs from projection")
            if self.projection.state is not ProductionReadinessAttestationStateV2.ACTIVE and (
                self.evidence_class is not ProductionAttestationEvidenceClassV2.PRODUCTION_VERIFIED
                or self.private_request_closure_sha256 is None
                or self.private_issuer_closure_sha256 is None
                or self.private_attestation_closure_sha256 is None
            ):
                raise ValueError("invalidated attestation closure is incomplete")
        expected = _result_facts(
            prerequisite=self.prerequisite,
            evidence_class=self.evidence_class,
            projection=self.projection,
            invalidation_record=self.invalidation_record,
            private_request=self.private_request_closure_sha256,
            private_issuer=self.private_issuer_closure_sha256,
            private_attestation=self.private_attestation_closure_sha256,
        )
        observed = (
            self.outcome,
            self.reason_codes,
            self.valid_until,
            self.active_attestation,
        )
        if observed != expected:
            raise ValueError("attestation result outcome differs from authority")
        if (
            self.passed_gate_count,
            self.failed_gate_count,
            self.pending_gate_count,
        ) != (
            self.prerequisite.passed_gate_count,
            self.prerequisite.failed_gate_count,
            self.prerequisite.pending_gate_count,
        ):
            raise ValueError("attestation result counts are not derived")
        gate_passed = tuple(
            item.outcome is ProductionReadinessGateOutcomeV2.PASSED
            for item in self.prerequisite.gate_assessments
        )
        if (
            self.satisfies_sc_010,
            self.satisfies_sc_011,
            self.satisfies_sc_012,
            self.satisfies_sc_013,
            self.satisfies_sc_014,
        ) != gate_passed:
            raise ValueError("SC authority differs from gate assessments")
        if self.evidence_class is ProductionAttestationEvidenceClassV2.REPOSITORY_PENDING_ONLY and any(
            value is not None
            for value in (
                self.projection,
                self.invalidation_record,
                self.private_request_closure_sha256,
                self.private_issuer_closure_sha256,
                self.private_attestation_closure_sha256,
            )
        ):
            raise ValueError("repository pending result carries private authority")
        refs = self.refs
        _require_audit(self.audit, refs, "production attestation result")
        _validate_identity(
            self.result_id,
            self.result_sha256,
            "production-readiness-attestation-result",
            production_readiness_attestation_result_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        policy_ref: ObjectRef,
        version_set: ProductionReadinessVersionSetV2,
        prerequisite: ProductionReadinessAttestationPrerequisiteV2,
        evidence_class: ProductionAttestationEvidenceClassV2,
        attestation_series_id: str,
        attestation_version: int,
        previous_result_ref: ObjectRef | None,
        projection: ProductionReadinessAttestationProjectionV2 | None,
        invalidation_record: ProductionReadinessInvalidationRecordV2 | None,
        private_request_closure_sha256: str | None,
        private_issuer_closure_sha256: str | None,
        private_attestation_closure_sha256: str | None,
        audit: ContractAudit,
    ) -> ProductionReadinessAttestationResultV2:
        outcome, reasons, valid_until, active = _result_facts(
            prerequisite=prerequisite,
            evidence_class=evidence_class,
            projection=projection,
            invalidation_record=invalidation_record,
            private_request=private_request_closure_sha256,
            private_issuer=private_issuer_closure_sha256,
            private_attestation=private_attestation_closure_sha256,
        )
        refs = (
            policy_ref,
            version_set.to_ref(),
            prerequisite.to_ref(),
            *((previous_result_ref,) if previous_result_ref is not None else ()),
            *((projection.to_ref(),) if projection is not None else ()),
            *((invalidation_record.to_ref(),) if invalidation_record is not None else ()),
        )
        passed = tuple(
            item.outcome is ProductionReadinessGateOutcomeV2.PASSED for item in prerequisite.gate_assessments
        )
        value = cls(
            result_id="production-readiness-attestation-result://pending",
            policy_ref=policy_ref,
            version_set=version_set,
            prerequisite=prerequisite,
            evidence_class=evidence_class,
            attestation_series_id=attestation_series_id,
            attestation_version=attestation_version,
            previous_result_ref=previous_result_ref,
            projection=projection,
            invalidation_record=invalidation_record,
            private_request_closure_sha256=private_request_closure_sha256,
            private_issuer_closure_sha256=private_issuer_closure_sha256,
            private_attestation_closure_sha256=private_attestation_closure_sha256,
            outcome=outcome,
            reason_codes=reasons,
            passed_gate_count=prerequisite.passed_gate_count,
            failed_gate_count=prerequisite.failed_gate_count,
            pending_gate_count=prerequisite.pending_gate_count,
            valid_until=valid_until,
            satisfies_sc_010=passed[0],
            satisfies_sc_011=passed[1],
            satisfies_sc_012=passed[2],
            satisfies_sc_013=passed[3],
            satisfies_sc_014=passed[4],
            active_attestation=active,
            result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "result_id",
            "result_sha256",
            "production-readiness-attestation-result",
            production_readiness_attestation_result_v2_carried_sha256(value),
        )

    @property
    def refs(self) -> tuple[ObjectRef, ...]:
        return (
            self.policy_ref,
            self.version_set.to_ref(),
            self.prerequisite.to_ref(),
            *((self.previous_result_ref,) if self.previous_result_ref is not None else ()),
            *((self.projection.to_ref(),) if self.projection is not None else ()),
            *((self.invalidation_record.to_ref(),) if self.invalidation_record is not None else ()),
        )

    def to_ref(self) -> ObjectRef:
        return production_readiness_attestation_result_v2_ref(self)


def production_readiness_attestation_policy_v2_carried_sha256(
    value: ProductionReadinessAttestationPolicyV2,
) -> str:
    return _carried(value, {"policy_id", "policy_sha256", "audit"})


def production_readiness_version_set_v2_carried_sha256(
    value: ProductionReadinessVersionSetV2,
) -> str:
    return _carried(value, {"version_set_id", "version_set_sha256", "audit"})


def production_readiness_gate_assessment_v2_carried_sha256(
    value: ProductionReadinessGateAssessmentV2,
) -> str:
    return _carried(value, {"assessment_id", "assessment_sha256"})


def production_readiness_attestation_prerequisite_v2_carried_sha256(
    value: ProductionReadinessAttestationPrerequisiteV2,
) -> str:
    return _carried(value, {"prerequisite_id", "prerequisite_sha256", "audit"})


def production_readiness_invalidation_policy_v2_carried_sha256(
    value: ProductionReadinessInvalidationPolicyV2,
) -> str:
    return _carried(value, {"policy_id", "policy_sha256", "audit"})


def production_readiness_invalidation_record_v2_carried_sha256(
    value: ProductionReadinessInvalidationRecordV2,
) -> str:
    return _carried(value, {"record_id", "record_sha256", "audit"})


def production_readiness_attestation_projection_v2_carried_sha256(
    value: ProductionReadinessAttestationProjectionV2,
) -> str:
    return _carried(value, {"projection_id", "projection_sha256", "audit"})


def production_readiness_attestation_result_v2_carried_sha256(
    value: ProductionReadinessAttestationResultV2,
) -> str:
    return _carried(value, {"result_id", "result_sha256", "audit"})


def production_readiness_attestation_policy_v2_ref(
    value: ProductionReadinessAttestationPolicyV2,
) -> ObjectRef:
    validate_production_readiness_attestation_policy_v2_identity(value)
    return _ref(
        "production-readiness-attestation-policy",
        value.policy_id,
        "v2",
        value.policy_sha256,
    )


def production_readiness_version_set_v2_ref(
    value: ProductionReadinessVersionSetV2,
) -> ObjectRef:
    validate_production_readiness_version_set_v2_identity(value)
    return _ref(
        "production-readiness-version-set",
        value.version_set_id,
        "v2",
        value.version_set_sha256,
    )


def production_readiness_gate_assessment_v2_ref(
    value: ProductionReadinessGateAssessmentV2,
) -> ObjectRef:
    validate_production_readiness_gate_assessment_v2_identity(value)
    return _ref(
        "production-readiness-gate-assessment",
        value.assessment_id,
        "v2",
        value.assessment_sha256,
    )


def production_readiness_attestation_prerequisite_v2_ref(
    value: ProductionReadinessAttestationPrerequisiteV2,
) -> ObjectRef:
    validate_production_readiness_attestation_prerequisite_v2_identity(value)
    return _ref(
        "production-readiness-attestation-prerequisite",
        value.prerequisite_id,
        "v2",
        value.prerequisite_sha256,
    )


def production_readiness_invalidation_policy_v2_ref(
    value: ProductionReadinessInvalidationPolicyV2,
) -> ObjectRef:
    validate_production_readiness_invalidation_policy_v2_identity(value)
    return _ref(
        "production-readiness-invalidation-policy",
        value.policy_id,
        "v2",
        value.policy_sha256,
    )


def production_readiness_invalidation_record_v2_ref(
    value: ProductionReadinessInvalidationRecordV2,
) -> ObjectRef:
    validate_production_readiness_invalidation_record_v2_identity(value)
    return _ref(
        "production-readiness-invalidation-record",
        value.record_id,
        "v2",
        value.record_sha256,
    )


def production_readiness_attestation_projection_v2_ref(
    value: ProductionReadinessAttestationProjectionV2,
) -> ObjectRef:
    validate_production_readiness_attestation_projection_v2_identity(value)
    return _ref(
        "production-readiness-attestation-projection",
        value.projection_id,
        "v2",
        value.projection_sha256,
    )


def production_readiness_attestation_result_v2_ref(
    value: ProductionReadinessAttestationResultV2,
) -> ObjectRef:
    validate_production_readiness_attestation_result_v2_identity(value)
    return _ref(
        "production-readiness-attestation-result",
        value.result_id,
        "v2",
        value.result_sha256,
    )


def validate_production_readiness_attestation_policy_v2_identity(
    value: ProductionReadinessAttestationPolicyV2,
) -> None:
    _validate_identity(
        value.policy_id,
        value.policy_sha256,
        "production-readiness-attestation-policy",
        production_readiness_attestation_policy_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_production_readiness_version_set_v2_identity(
    value: ProductionReadinessVersionSetV2,
) -> None:
    _validate_identity(
        value.version_set_id,
        value.version_set_sha256,
        "production-readiness-version-set",
        production_readiness_version_set_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_production_readiness_gate_assessment_v2_identity(
    value: ProductionReadinessGateAssessmentV2,
) -> None:
    _validate_identity(
        value.assessment_id,
        value.assessment_sha256,
        "production-readiness-gate-assessment",
        production_readiness_gate_assessment_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_production_readiness_attestation_prerequisite_v2_identity(
    value: ProductionReadinessAttestationPrerequisiteV2,
) -> None:
    _validate_identity(
        value.prerequisite_id,
        value.prerequisite_sha256,
        "production-readiness-attestation-prerequisite",
        production_readiness_attestation_prerequisite_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_production_readiness_invalidation_policy_v2_identity(
    value: ProductionReadinessInvalidationPolicyV2,
) -> None:
    _validate_identity(
        value.policy_id,
        value.policy_sha256,
        "production-readiness-invalidation-policy",
        production_readiness_invalidation_policy_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_production_readiness_invalidation_record_v2_identity(
    value: ProductionReadinessInvalidationRecordV2,
) -> None:
    _validate_identity(
        value.record_id,
        value.record_sha256,
        "production-readiness-invalidation-record",
        production_readiness_invalidation_record_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_production_readiness_attestation_projection_v2_identity(
    value: ProductionReadinessAttestationProjectionV2,
) -> None:
    _validate_identity(
        value.projection_id,
        value.projection_sha256,
        "production-readiness-attestation-projection",
        production_readiness_attestation_projection_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_production_readiness_attestation_result_v2_identity(
    value: ProductionReadinessAttestationResultV2,
) -> None:
    _validate_identity(
        value.result_id,
        value.result_sha256,
        "production-readiness-attestation-result",
        production_readiness_attestation_result_v2_carried_sha256(value),
        allow_pending=False,
    )


def _gate_counts(
    gates: tuple[ProductionReadinessGateAssessmentV2, ...],
) -> tuple[int, int, int]:
    return (
        sum(item.outcome is ProductionReadinessGateOutcomeV2.PASSED for item in gates),
        sum(item.outcome is ProductionReadinessGateOutcomeV2.FAILED for item in gates),
        sum(item.outcome is ProductionReadinessGateOutcomeV2.PENDING for item in gates),
    )


def _prerequisite_outcome(
    gates: tuple[ProductionReadinessGateAssessmentV2, ...],
) -> tuple[
    ProductionReadinessAttestationOutcomeV2,
    tuple[ProductionReadinessAttestationReasonCodeV2, ...],
    datetime | None,
]:
    reasons = tuple(
        sorted(
            {
                reason
                for gate in gates
                for reason in gate.reason_codes
                if reason is not ProductionReadinessAttestationReasonCodeV2.NONE
            },
            key=lambda item: item.value,
        )
    )
    if any(item.outcome is ProductionReadinessGateOutcomeV2.FAILED for item in gates):
        return (
            ProductionReadinessAttestationOutcomeV2.REJECTED,
            reasons or (ProductionReadinessAttestationReasonCodeV2.GATE_FAILED,),
            None,
        )
    if any(item.outcome is ProductionReadinessGateOutcomeV2.PENDING for item in gates):
        return (
            ProductionReadinessAttestationOutcomeV2.ATTESTATION_PENDING,
            reasons,
            None,
        )
    return (
        ProductionReadinessAttestationOutcomeV2.ISSUED,
        (ProductionReadinessAttestationReasonCodeV2.NONE,),
        min(item.valid_until for item in gates if item.valid_until is not None),
    )


def _result_facts(
    *,
    prerequisite: ProductionReadinessAttestationPrerequisiteV2,
    evidence_class: ProductionAttestationEvidenceClassV2,
    projection: ProductionReadinessAttestationProjectionV2 | None,
    invalidation_record: ProductionReadinessInvalidationRecordV2 | None,
    private_request: str | None,
    private_issuer: str | None,
    private_attestation: str | None,
) -> tuple[
    ProductionReadinessAttestationOutcomeV2,
    tuple[ProductionReadinessAttestationReasonCodeV2, ...],
    datetime | None,
    bool,
]:
    if projection is not None and projection.state is ProductionReadinessAttestationStateV2.INVALIDATED:
        if invalidation_record is None:
            raise ValueError("invalidated projection requires invalidation record")
        return (
            ProductionReadinessAttestationOutcomeV2.INVALIDATED,
            (ProductionReadinessAttestationReasonCodeV2.ATTESTATION_INVALIDATED,),
            None,
            False,
        )
    if projection is not None and projection.state is ProductionReadinessAttestationStateV2.EXPIRED:
        if invalidation_record is None:
            raise ValueError("expired projection requires invalidation record")
        return (
            ProductionReadinessAttestationOutcomeV2.EXPIRED,
            (ProductionReadinessAttestationReasonCodeV2.ATTESTATION_EXPIRED,),
            None,
            False,
        )
    if prerequisite.outcome is ProductionReadinessAttestationOutcomeV2.REJECTED:
        if projection is not None or private_attestation is not None:
            raise ValueError("rejected attestation result carries issued authority")
        return (
            ProductionReadinessAttestationOutcomeV2.REJECTED,
            prerequisite.reason_codes,
            None,
            False,
        )
    if prerequisite.outcome is ProductionReadinessAttestationOutcomeV2.ATTESTATION_PENDING:
        if projection is not None or private_attestation is not None:
            raise ValueError("pending attestation result carries issued authority")
        return (
            ProductionReadinessAttestationOutcomeV2.ATTESTATION_PENDING,
            prerequisite.reason_codes,
            None,
            False,
        )
    if evidence_class is ProductionAttestationEvidenceClassV2.MECHANISM_VALIDATION_ONLY:
        if projection is not None or private_attestation is not None:
            raise ValueError("mechanism result cannot carry attestation")
        return (
            ProductionReadinessAttestationOutcomeV2.ATTESTATION_PENDING,
            (ProductionReadinessAttestationReasonCodeV2.MECHANISM_ONLY,),
            None,
            False,
        )
    if evidence_class is ProductionAttestationEvidenceClassV2.REPOSITORY_PENDING_ONLY:
        raise ValueError("repository pending evidence cannot pass every gate")
    if (
        projection is None
        or projection.state is not ProductionReadinessAttestationStateV2.ACTIVE
        or invalidation_record is not None
        or private_request is None
        or private_issuer is None
        or private_attestation is None
    ):
        raise ValueError("production issuance closure is incomplete")
    return (
        ProductionReadinessAttestationOutcomeV2.ISSUED,
        (ProductionReadinessAttestationReasonCodeV2.NONE,),
        projection.valid_until,
        True,
    )


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")


def _require_sorted_unique_refs(
    values: tuple[ObjectRef, ...],
    field_name: str,
) -> None:
    keys = tuple(_ref_key(value) for value in values)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} must be sorted and unique")


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return audit.model_copy(update={"input_refs": _sorted_refs(refs)})


def _require_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != _sorted_refs(refs):
        raise ValueError(f"{label} audit refs are incomplete")


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    unique = {_ref_key(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


def _require_ref(
    value: ObjectRef,
    object_type: str,
    object_version: str,
    field_name: str,
) -> None:
    if value.object_type != object_type or value.object_version != object_version:
        raise ValueError(f"{field_name} must reference {object_type} {object_version}")


def _ref(
    object_type: str,
    object_id: str,
    version: str,
    digest: str,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version=version,
        object_sha256=digest,
    )


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _carried(
    value: ContractModelV2,
    exclude: set[str],
) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(
                value.model_dump(
                    mode="python",
                    exclude=exclude,
                    exclude_none=False,
                )
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _finalize[ModelT: ContractModelV2](
    value: ModelT,
    id_field: str,
    hash_field: str,
    prefix: str,
    digest: str,
) -> ModelT:
    return value.model_copy(
        update={
            id_field: f"{prefix}://sha256/{digest}",
            hash_field: digest,
        }
    )


def _validate_identity(
    object_id: str,
    object_sha256: str,
    prefix: str,
    observed: str,
    *,
    allow_pending: bool = True,
) -> None:
    if allow_pending and object_id == f"{prefix}://pending" and object_sha256 == "0" * 64:
        return
    if object_sha256 != observed or object_id != f"{prefix}://sha256/{observed}":
        raise ValueError(f"{prefix} identity is stale")


__all__ = [
    "PRODUCTION_ATTESTATION_GATE_ORDER",
    "PRODUCTION_ATTESTATION_INVALIDATION_TRIGGER_ORDER",
    "PRODUCTION_READINESS_ATTESTATION_POLICY_VERSION",
    "ProductionAttestationEvidenceClassV2",
    "ProductionReadinessAttestationOutcomeV2",
    "ProductionReadinessAttestationPolicyV2",
    "ProductionReadinessAttestationPrerequisiteV2",
    "ProductionReadinessAttestationProjectionV2",
    "ProductionReadinessAttestationReasonCodeV2",
    "ProductionReadinessAttestationResultV2",
    "ProductionReadinessAttestationStateV2",
    "ProductionReadinessGateAssessmentV2",
    "ProductionReadinessGateOutcomeV2",
    "ProductionReadinessGateV2",
    "ProductionReadinessInvalidationPolicyV2",
    "ProductionReadinessInvalidationRecordV2",
    "ProductionReadinessInvalidationTriggerV2",
    "ProductionReadinessVersionSetV2",
    "production_readiness_attestation_policy_v2_carried_sha256",
    "production_readiness_attestation_policy_v2_ref",
    "production_readiness_attestation_prerequisite_v2_carried_sha256",
    "production_readiness_attestation_prerequisite_v2_ref",
    "production_readiness_attestation_projection_v2_carried_sha256",
    "production_readiness_attestation_projection_v2_ref",
    "production_readiness_attestation_result_v2_carried_sha256",
    "production_readiness_attestation_result_v2_ref",
    "production_readiness_gate_assessment_v2_carried_sha256",
    "production_readiness_gate_assessment_v2_ref",
    "production_readiness_invalidation_policy_v2_carried_sha256",
    "production_readiness_invalidation_policy_v2_ref",
    "production_readiness_invalidation_record_v2_carried_sha256",
    "production_readiness_invalidation_record_v2_ref",
    "production_readiness_version_set_v2_carried_sha256",
    "production_readiness_version_set_v2_ref",
    "validate_production_readiness_attestation_policy_v2_identity",
    "validate_production_readiness_attestation_prerequisite_v2_identity",
    "validate_production_readiness_attestation_projection_v2_identity",
    "validate_production_readiness_attestation_result_v2_identity",
    "validate_production_readiness_gate_assessment_v2_identity",
    "validate_production_readiness_invalidation_policy_v2_identity",
    "validate_production_readiness_invalidation_record_v2_identity",
    "validate_production_readiness_version_set_v2_identity",
]
