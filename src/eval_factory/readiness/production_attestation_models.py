from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.concurrency_experiment_v2 import (
    ConcurrencyExperimentReportV2,
    validate_concurrency_experiment_report_v2_identity,
)
from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.label_quality_v2 import (
    LabelQualityEvaluationReportV2,
    validate_label_quality_evaluation_report_v2_identity,
)
from eval_factory.contracts.production_attestation_v2 import (
    PRODUCTION_ATTESTATION_GATE_ORDER,
    ProductionReadinessAttestationPrerequisiteV2,
    ProductionReadinessGateV2,
    ProductionReadinessVersionSetV2,
    validate_production_readiness_attestation_prerequisite_v2_identity,
    validate_production_readiness_version_set_v2_identity,
)
from eval_factory.contracts.production_readiness_review_v2 import (
    ProductionReadinessReviewReportV2,
    validate_production_readiness_review_report_v2_identity,
)
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityReportV2,
    validate_real_trace_stability_report_v2_identity,
)

ATTESTATION_ISSUER_VERIFICATION_METHOD: Literal["DETACHED_REGISTRY_PROOF_V1"] = "DETACHED_REGISTRY_PROOF_V1"


class TrustedAttestationIssuerV1(ContractModelV2):
    schema_version: Literal["eval-factory/trusted-attestation-issuer/private-v1"] = (
        "eval-factory/trusted-attestation-issuer/private-v1"
    )
    issuer_principal: Identifier
    issuer_organization: Identifier
    issuer_role: Identifier
    issuance_scope: Literal["PRODUCTION_READINESS_ATTESTATION"]
    issuance_scope_version: str = Field(min_length=1, max_length=128)
    verification_method: Literal["DETACHED_REGISTRY_PROOF_V1"] = ATTESTATION_ISSUER_VERIFICATION_METHOD
    valid_from: datetime
    valid_until: datetime
    registry_evidence_sha256: Sha256

    @model_validator(mode="after")
    def validate_issuer(self) -> Self:
        _require_aware(self.valid_from, "issuer valid_from")
        _require_aware(self.valid_until, "issuer valid_until")
        if self.valid_until <= self.valid_from:
            raise ValueError("issuer validity window is invalid")
        return self


class TrustedAttestationIssuerRegistryV1(ContractModelV2):
    schema_version: Literal["eval-factory/trusted-attestation-issuer-registry/private-v1"] = (
        "eval-factory/trusted-attestation-issuer-registry/private-v1"
    )
    registry_id: Identifier
    registry_version: str = Field(min_length=1, max_length=128)
    project_owner_principal: Identifier
    issuers: tuple[TrustedAttestationIssuerV1, ...] = Field(
        min_length=2,
        max_length=100,
    )
    minimum_independent_issuer_count: int = Field(ge=2, le=100)
    registry_proof_commitment_sha256: Sha256
    valid_from: datetime
    valid_until: datetime
    registry_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_registry(self) -> Self:
        principals = tuple(item.issuer_principal for item in self.issuers)
        if principals != tuple(sorted(principals)) or len(principals) != len(set(principals)):
            raise ValueError("issuer registry principals must be sorted and unique")
        if len(self.issuers) < self.minimum_independent_issuer_count:
            raise ValueError("issuer registry has insufficient independent authority")
        if any(item.issuer_principal == self.project_owner_principal for item in self.issuers):
            raise ValueError("project owner cannot be an attestation issuer")
        _require_aware(self.valid_from, "issuer registry valid_from")
        _require_aware(self.valid_until, "issuer registry valid_until")
        if self.valid_until <= self.valid_from:
            raise ValueError("issuer registry validity window is invalid")
        _require_audit(self.audit, (), "trusted attestation issuer registry")
        _validate_identity(
            self.registry_id,
            self.registry_sha256,
            "attestation-issuer-registry",
            trusted_attestation_issuer_registry_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        registry_version: str,
        project_owner_principal: str,
        issuers: tuple[TrustedAttestationIssuerV1, ...],
        minimum_independent_issuer_count: int,
        registry_proof_commitment_sha256: str,
        valid_from: datetime,
        valid_until: datetime,
        audit: ContractAudit,
    ) -> TrustedAttestationIssuerRegistryV1:
        ordered = tuple(sorted(issuers, key=lambda item: item.issuer_principal))
        value = cls(
            registry_id="attestation-issuer-registry://pending",
            registry_version=registry_version,
            project_owner_principal=project_owner_principal,
            issuers=ordered,
            minimum_independent_issuer_count=minimum_independent_issuer_count,
            registry_proof_commitment_sha256=registry_proof_commitment_sha256,
            valid_from=valid_from,
            valid_until=valid_until,
            registry_sha256="0" * 64,
            audit=_safe_audit(audit, ()),
        )
        return _finalize(
            value,
            "registry_id",
            "registry_sha256",
            "attestation-issuer-registry",
            trusted_attestation_issuer_registry_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return trusted_attestation_issuer_registry_v1_ref(self)


class AttestationIssuanceProofV1(ContractModelV2):
    schema_version: Literal["eval-factory/attestation-issuance-proof/private-v1"] = (
        "eval-factory/attestation-issuance-proof/private-v1"
    )
    proof_id: Identifier
    issuer_principal: Identifier
    issuer_organization: Identifier
    issuer_role: Identifier
    issuance_scope: Literal["PRODUCTION_READINESS_ATTESTATION"]
    issuance_scope_version: str = Field(min_length=1, max_length=128)
    verification_method: Literal["DETACHED_REGISTRY_PROOF_V1"] = ATTESTATION_ISSUER_VERIFICATION_METHOD
    attestation_series_id: Identifier
    attestation_version: int = Field(ge=1, le=1_000_000)
    version_set_ref: ObjectRef
    prerequisite_ref: ObjectRef
    evidence_refs: tuple[ObjectRef, ...] = Field(min_length=4, max_length=1_000)
    policy_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=1_000)
    decision: Literal["APPROVE", "REJECT"]
    proof_value: str = Field(min_length=16, max_length=16_384)
    proof_value_sha256: Sha256
    issued_at: datetime
    valid_from: datetime
    valid_until: datetime
    proof_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_proof(self) -> Self:
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
        _require_sorted_unique_refs(self.evidence_refs, "issuance evidence refs")
        _require_sorted_unique_refs(self.policy_refs, "issuance policy refs")
        if hashlib.sha256(self.proof_value.encode()).hexdigest() != (self.proof_value_sha256):
            raise ValueError("issuance proof value hash is stale")
        for value, label in (
            (self.issued_at, "issuance proof issued_at"),
            (self.valid_from, "issuance proof valid_from"),
            (self.valid_until, "issuance proof valid_until"),
        ):
            _require_aware(value, label)
        if self.valid_until <= self.valid_from or self.issued_at > self.valid_from:
            raise ValueError("issuance proof validity window is invalid")
        refs = (
            self.version_set_ref,
            self.prerequisite_ref,
            *self.evidence_refs,
            *self.policy_refs,
        )
        _require_audit(self.audit, refs, "attestation issuance proof")
        _validate_identity(
            self.proof_id,
            self.proof_sha256,
            "attestation-issuance-proof",
            attestation_issuance_proof_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        issuer: TrustedAttestationIssuerV1,
        attestation_series_id: str,
        attestation_version: int,
        version_set_ref: ObjectRef,
        prerequisite_ref: ObjectRef,
        evidence_refs: tuple[ObjectRef, ...],
        policy_refs: tuple[ObjectRef, ...],
        decision: Literal["APPROVE", "REJECT"],
        proof_value: str,
        issued_at: datetime,
        valid_from: datetime,
        valid_until: datetime,
        audit: ContractAudit,
    ) -> AttestationIssuanceProofV1:
        evidence = _sorted_refs(evidence_refs)
        policies = _sorted_refs(policy_refs)
        refs = (version_set_ref, prerequisite_ref, *evidence, *policies)
        value = cls(
            proof_id="attestation-issuance-proof://pending",
            issuer_principal=issuer.issuer_principal,
            issuer_organization=issuer.issuer_organization,
            issuer_role=issuer.issuer_role,
            issuance_scope=issuer.issuance_scope,
            issuance_scope_version=issuer.issuance_scope_version,
            verification_method=issuer.verification_method,
            attestation_series_id=attestation_series_id,
            attestation_version=attestation_version,
            version_set_ref=version_set_ref,
            prerequisite_ref=prerequisite_ref,
            evidence_refs=evidence,
            policy_refs=policies,
            decision=decision,
            proof_value=proof_value,
            proof_value_sha256=hashlib.sha256(proof_value.encode()).hexdigest(),
            issued_at=issued_at,
            valid_from=valid_from,
            valid_until=valid_until,
            proof_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "proof_id",
            "proof_sha256",
            "attestation-issuance-proof",
            attestation_issuance_proof_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return attestation_issuance_proof_v1_ref(self)


class AttestationIssuanceBundleV1(ContractModelV2):
    schema_version: Literal["eval-factory/attestation-issuance-bundle/private-v1"] = (
        "eval-factory/attestation-issuance-bundle/private-v1"
    )
    bundle_id: Identifier
    attestation_series_id: Identifier
    attestation_version: int = Field(ge=1, le=1_000_000)
    proofs: tuple[AttestationIssuanceProofV1, ...] = Field(
        min_length=1,
        max_length=100,
    )
    bundle_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        principals = tuple(item.issuer_principal for item in self.proofs)
        if principals != tuple(sorted(principals)) or len(principals) != len(set(principals)):
            raise ValueError("issuance bundle principals must be sorted and unique")
        if any(
            item.attestation_series_id != self.attestation_series_id
            or item.attestation_version != self.attestation_version
            for item in self.proofs
        ):
            raise ValueError("issuance proof version differs from bundle")
        refs = tuple(item.to_ref() for item in self.proofs)
        _require_audit(self.audit, refs, "attestation issuance bundle")
        _validate_identity(
            self.bundle_id,
            self.bundle_sha256,
            "attestation-issuance-bundle",
            attestation_issuance_bundle_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        attestation_series_id: str,
        attestation_version: int,
        proofs: tuple[AttestationIssuanceProofV1, ...],
        audit: ContractAudit,
    ) -> AttestationIssuanceBundleV1:
        ordered = tuple(sorted(proofs, key=lambda item: item.issuer_principal))
        refs = tuple(item.to_ref() for item in ordered)
        value = cls(
            bundle_id="attestation-issuance-bundle://pending",
            attestation_series_id=attestation_series_id,
            attestation_version=attestation_version,
            proofs=ordered,
            bundle_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "bundle_id",
            "bundle_sha256",
            "attestation-issuance-bundle",
            attestation_issuance_bundle_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return attestation_issuance_bundle_v1_ref(self)


class ProductionReadinessSC014ClosureV1(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-sc-014-closure/private-v1"] = (
        "eval-factory/production-readiness-sc-014-closure/private-v1"
    )
    closure_id: Identifier
    statistical_gate_open_count: int = Field(ge=0)
    open_p0_count: int = Field(ge=0)
    open_p1_count: int = Field(ge=0)
    open_non_waivable_count: int = Field(ge=0)
    required_checkpoint_open_count: int = Field(ge=0)
    release_integrity_open_count: int = Field(ge=0)
    evidence_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=10_000)
    lifecycle_approval_ref: ObjectRef
    incident_route_ref: ObjectRef
    evaluated_at: datetime
    valid_until: datetime
    closure_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_closure(self) -> Self:
        _require_sorted_unique_refs(self.evidence_refs, "SC-014 evidence refs")
        _require_ref(
            self.lifecycle_approval_ref,
            "data-lifecycle-approval",
            "v1",
            "lifecycle_approval_ref",
        )
        _require_ref(
            self.incident_route_ref,
            "incident-route",
            "v1",
            "incident_route_ref",
        )
        _require_aware(self.evaluated_at, "SC-014 evaluated_at")
        _require_aware(self.valid_until, "SC-014 valid_until")
        if self.valid_until <= self.evaluated_at:
            raise ValueError("SC-014 closure validity is invalid")
        refs = (
            *self.evidence_refs,
            self.lifecycle_approval_ref,
            self.incident_route_ref,
        )
        _require_audit(self.audit, refs, "SC-014 closure")
        _validate_identity(
            self.closure_id,
            self.closure_sha256,
            "production-readiness-sc-014-closure",
            production_readiness_sc014_closure_v1_carried_sha256(self),
        )
        return self

    @property
    def open_count(self) -> int:
        return (
            self.statistical_gate_open_count
            + self.open_p0_count
            + self.open_p1_count
            + self.open_non_waivable_count
            + self.required_checkpoint_open_count
            + self.release_integrity_open_count
        )

    @classmethod
    def create(
        cls,
        *,
        statistical_gate_open_count: int,
        open_p0_count: int,
        open_p1_count: int,
        open_non_waivable_count: int,
        required_checkpoint_open_count: int,
        release_integrity_open_count: int,
        evidence_refs: tuple[ObjectRef, ...],
        lifecycle_approval_ref: ObjectRef,
        incident_route_ref: ObjectRef,
        evaluated_at: datetime,
        valid_until: datetime,
        audit: ContractAudit,
    ) -> ProductionReadinessSC014ClosureV1:
        evidence = _sorted_refs(evidence_refs)
        refs = (*evidence, lifecycle_approval_ref, incident_route_ref)
        value = cls(
            closure_id="production-readiness-sc-014-closure://pending",
            statistical_gate_open_count=statistical_gate_open_count,
            open_p0_count=open_p0_count,
            open_p1_count=open_p1_count,
            open_non_waivable_count=open_non_waivable_count,
            required_checkpoint_open_count=required_checkpoint_open_count,
            release_integrity_open_count=release_integrity_open_count,
            evidence_refs=evidence,
            lifecycle_approval_ref=lifecycle_approval_ref,
            incident_route_ref=incident_route_ref,
            evaluated_at=evaluated_at,
            valid_until=valid_until,
            closure_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "closure_id",
            "closure_sha256",
            "production-readiness-sc-014-closure",
            production_readiness_sc014_closure_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return production_readiness_sc014_closure_v1_ref(self)


class ProductionReadinessAttestationRequestV1(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-attestation-request/private-v1"] = (
        "eval-factory/production-readiness-attestation-request/private-v1"
    )
    request_id: Identifier
    attestation_series_id: Identifier
    attestation_version: int = Field(ge=1, le=1_000_000)
    previous_result_ref: ObjectRef | None = None
    previous_projection_ref: ObjectRef | None = None
    label_quality_report: LabelQualityEvaluationReportV2
    real_trace_stability_report: RealTraceStabilityReportV2
    concurrency_experiment_report: ConcurrencyExperimentReportV2
    production_readiness_review_report: ProductionReadinessReviewReportV2
    version_set: ProductionReadinessVersionSetV2
    expected_prerequisite: ProductionReadinessAttestationPrerequisiteV2
    sc_014_closure: ProductionReadinessSC014ClosureV1
    issuance_bundle: AttestationIssuanceBundleV1
    requested_valid_from: datetime
    requested_valid_until: datetime
    evaluated_at: datetime
    request_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        validate_label_quality_evaluation_report_v2_identity(self.label_quality_report)
        validate_real_trace_stability_report_v2_identity(self.real_trace_stability_report)
        validate_concurrency_experiment_report_v2_identity(self.concurrency_experiment_report)
        validate_production_readiness_review_report_v2_identity(self.production_readiness_review_report)
        validate_production_readiness_version_set_v2_identity(self.version_set)
        validate_production_readiness_attestation_prerequisite_v2_identity(self.expected_prerequisite)
        if self.attestation_version == 1:
            if self.previous_result_ref is not None or self.previous_projection_ref is not None:
                raise ValueError("attestation request version 1 cannot have predecessor")
        else:
            if self.previous_result_ref is None or self.previous_projection_ref is None:
                raise ValueError("later attestation request requires predecessor")
            _require_ref(
                self.previous_result_ref,
                "production-readiness-attestation-result",
                "v2",
                "previous_result_ref",
            )
            _require_ref(
                self.previous_projection_ref,
                "production-readiness-attestation-projection",
                "v2",
                "previous_projection_ref",
            )
        if (
            self.issuance_bundle.attestation_series_id != self.attestation_series_id
            or self.issuance_bundle.attestation_version != self.attestation_version
        ):
            raise ValueError("issuance bundle differs from attestation request")
        for value, label in (
            (self.requested_valid_from, "request valid_from"),
            (self.requested_valid_until, "request valid_until"),
            (self.evaluated_at, "request evaluated_at"),
        ):
            _require_aware(value, label)
        if (
            self.requested_valid_until <= self.requested_valid_from
            or not self.requested_valid_from <= self.evaluated_at < self.requested_valid_until
        ):
            raise ValueError("attestation request validity window is invalid")
        refs = self.refs
        _require_audit(self.audit, refs, "production attestation request")
        _validate_identity(
            self.request_id,
            self.request_sha256,
            "production-readiness-attestation-request",
            production_readiness_attestation_request_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        attestation_series_id: str,
        attestation_version: int,
        previous_result_ref: ObjectRef | None,
        previous_projection_ref: ObjectRef | None,
        label_quality_report: LabelQualityEvaluationReportV2,
        real_trace_stability_report: RealTraceStabilityReportV2,
        concurrency_experiment_report: ConcurrencyExperimentReportV2,
        production_readiness_review_report: ProductionReadinessReviewReportV2,
        version_set: ProductionReadinessVersionSetV2,
        expected_prerequisite: ProductionReadinessAttestationPrerequisiteV2,
        sc_014_closure: ProductionReadinessSC014ClosureV1,
        issuance_bundle: AttestationIssuanceBundleV1,
        requested_valid_from: datetime,
        requested_valid_until: datetime,
        evaluated_at: datetime,
        audit: ContractAudit,
    ) -> ProductionReadinessAttestationRequestV1:
        refs = _request_refs(
            previous_result_ref=previous_result_ref,
            previous_projection_ref=previous_projection_ref,
            label_quality_report=label_quality_report,
            real_trace_stability_report=real_trace_stability_report,
            concurrency_experiment_report=concurrency_experiment_report,
            production_readiness_review_report=production_readiness_review_report,
            version_set=version_set,
            expected_prerequisite=expected_prerequisite,
            sc_014_closure=sc_014_closure,
            issuance_bundle=issuance_bundle,
        )
        value = cls(
            request_id="production-readiness-attestation-request://pending",
            attestation_series_id=attestation_series_id,
            attestation_version=attestation_version,
            previous_result_ref=previous_result_ref,
            previous_projection_ref=previous_projection_ref,
            label_quality_report=label_quality_report,
            real_trace_stability_report=real_trace_stability_report,
            concurrency_experiment_report=concurrency_experiment_report,
            production_readiness_review_report=production_readiness_review_report,
            version_set=version_set,
            expected_prerequisite=expected_prerequisite,
            sc_014_closure=sc_014_closure,
            issuance_bundle=issuance_bundle,
            requested_valid_from=requested_valid_from,
            requested_valid_until=requested_valid_until,
            evaluated_at=evaluated_at,
            request_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "request_id",
            "request_sha256",
            "production-readiness-attestation-request",
            production_readiness_attestation_request_v1_carried_sha256(value),
        )

    @property
    def refs(self) -> tuple[ObjectRef, ...]:
        return _request_refs(
            previous_result_ref=self.previous_result_ref,
            previous_projection_ref=self.previous_projection_ref,
            label_quality_report=self.label_quality_report,
            real_trace_stability_report=self.real_trace_stability_report,
            concurrency_experiment_report=self.concurrency_experiment_report,
            production_readiness_review_report=self.production_readiness_review_report,
            version_set=self.version_set,
            expected_prerequisite=self.expected_prerequisite,
            sc_014_closure=self.sc_014_closure,
            issuance_bundle=self.issuance_bundle,
        )

    def to_ref(self) -> ObjectRef:
        return production_readiness_attestation_request_v1_ref(self)


class RepositoryPendingAttestationPredecessorV1(ContractModelV2):
    schema_version: Literal["eval-factory/repository-pending-attestation-predecessor/private-v1"] = (
        "eval-factory/repository-pending-attestation-predecessor/private-v1"
    )
    stage: Literal["R8_02", "R8_04", "R8_06", "R8_07"]
    report_sha256: Sha256
    outcome: str = Field(min_length=1, max_length=128)
    claim_scope: str = Field(min_length=1, max_length=128)
    satisfies_stage_gate: Literal[False]


class RepositoryPendingAttestationEvidenceV1(ContractModelV2):
    schema_version: Literal["eval-factory/repository-pending-production-attestation-evidence/v1"] = (
        "eval-factory/repository-pending-production-attestation-evidence/v1"
    )
    evidence_class: Literal["REPOSITORY_PENDING_ONLY"]
    predecessors: tuple[RepositoryPendingAttestationPredecessorV1, ...] = Field(
        min_length=4,
        max_length=4,
    )
    pending_gates: tuple[ProductionReadinessGateV2, ...] = Field(
        min_length=5,
        max_length=5,
    )
    sc_014_closure_present: Literal[False]
    lifecycle_approval_present: Literal[False]
    incident_route_present: Literal[False]
    issuer_authority_present: Literal[False]
    frozen_attestation_present: Literal[False]
    authorizes_production_release: Literal[False]

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if tuple(item.stage for item in self.predecessors) != (
            "R8_02",
            "R8_04",
            "R8_06",
            "R8_07",
        ):
            raise ValueError("repository attestation predecessors are not canonical")
        if self.pending_gates != PRODUCTION_ATTESTATION_GATE_ORDER:
            raise ValueError("repository pending gate inventory is not canonical")
        return self


class ProductionReadinessAttestationAcceptedRunV1(ContractModelV2):
    schema_version: Literal["eval-factory/production-attestation-accepted-run/private-v1"] = (
        "eval-factory/production-attestation-accepted-run/private-v1"
    )
    accepted_run_id: Identifier
    acceptance_key: Identifier
    request_sha256: Sha256
    result_ref: ObjectRef
    registry_ref: ObjectRef | None = None
    request_ref: ObjectRef | None = None
    bundle_ref: ObjectRef | None = None
    frozen_attestation_ref: ObjectRef | None = None
    current_projection_ref: ObjectRef | None = None
    accepted_run_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        _require_ref(
            self.result_ref,
            "production-readiness-attestation-result",
            "v2",
            "result_ref",
        )
        request_values = (
            self.registry_ref,
            self.request_ref,
            self.bundle_ref,
        )
        issued_values = (
            self.frozen_attestation_ref,
            self.current_projection_ref,
        )
        if self.request_ref is None:
            if any(item is not None for item in (*request_values, *issued_values)):
                raise ValueError("pending repository acceptance carries private closure")
        else:
            if any(item is None for item in request_values):
                raise ValueError("accepted full request closure is incomplete")
            if any(item is not None for item in issued_values) and any(
                item is None for item in issued_values
            ):
                raise ValueError("accepted issued closure must be all-or-none")
            assert self.registry_ref is not None
            assert self.bundle_ref is not None
            _require_ref(
                self.registry_ref,
                "attestation-issuer-registry",
                "private-v1",
                "registry_ref",
            )
            _require_ref(
                self.request_ref,
                "production-readiness-attestation-request",
                "private-v1",
                "request_ref",
            )
            _require_ref(
                self.bundle_ref,
                "attestation-issuance-bundle",
                "private-v1",
                "bundle_ref",
            )
            if self.frozen_attestation_ref is not None:
                assert self.current_projection_ref is not None
                _require_ref(
                    self.frozen_attestation_ref,
                    "production-readiness-attestation",
                    "v1",
                    "frozen_attestation_ref",
                )
                _require_ref(
                    self.current_projection_ref,
                    "production-readiness-attestation-projection",
                    "v2",
                    "current_projection_ref",
                )
        refs = (
            self.result_ref,
            *(item for item in (*request_values, *issued_values) if item is not None),
        )
        _require_audit(self.audit, refs, "production attestation accepted run")
        _validate_identity(
            self.accepted_run_id,
            self.accepted_run_sha256,
            "production-attestation-accepted-run",
            production_readiness_attestation_accepted_run_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        acceptance_key: str,
        request_sha256: str,
        result_ref: ObjectRef,
        registry_ref: ObjectRef | None,
        request_ref: ObjectRef | None,
        bundle_ref: ObjectRef | None,
        frozen_attestation_ref: ObjectRef | None,
        current_projection_ref: ObjectRef | None,
        audit: ContractAudit,
    ) -> ProductionReadinessAttestationAcceptedRunV1:
        refs = (
            result_ref,
            *(
                item
                for item in (
                    registry_ref,
                    request_ref,
                    bundle_ref,
                    frozen_attestation_ref,
                    current_projection_ref,
                )
                if item is not None
            ),
        )
        value = cls(
            accepted_run_id="production-attestation-accepted-run://pending",
            acceptance_key=acceptance_key,
            request_sha256=request_sha256,
            result_ref=result_ref,
            registry_ref=registry_ref,
            request_ref=request_ref,
            bundle_ref=bundle_ref,
            frozen_attestation_ref=frozen_attestation_ref,
            current_projection_ref=current_projection_ref,
            accepted_run_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "accepted_run_id",
            "accepted_run_sha256",
            "production-attestation-accepted-run",
            production_readiness_attestation_accepted_run_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return production_readiness_attestation_accepted_run_v1_ref(self)


def trusted_attestation_issuer_registry_v1_carried_sha256(
    value: TrustedAttestationIssuerRegistryV1,
) -> str:
    return _carried(value, {"registry_id", "registry_sha256", "audit"})


def attestation_issuance_proof_v1_carried_sha256(
    value: AttestationIssuanceProofV1,
) -> str:
    return _carried(value, {"proof_id", "proof_sha256", "audit"})


def attestation_issuance_bundle_v1_carried_sha256(
    value: AttestationIssuanceBundleV1,
) -> str:
    return _carried(value, {"bundle_id", "bundle_sha256", "audit"})


def production_readiness_sc014_closure_v1_carried_sha256(
    value: ProductionReadinessSC014ClosureV1,
) -> str:
    return _carried(value, {"closure_id", "closure_sha256", "audit"})


def production_readiness_attestation_request_v1_carried_sha256(
    value: ProductionReadinessAttestationRequestV1,
) -> str:
    return _carried(value, {"request_id", "request_sha256", "audit"})


def production_readiness_attestation_accepted_run_v1_carried_sha256(
    value: ProductionReadinessAttestationAcceptedRunV1,
) -> str:
    return _carried(value, {"accepted_run_id", "accepted_run_sha256", "audit"})


def trusted_attestation_issuer_registry_v1_ref(
    value: TrustedAttestationIssuerRegistryV1,
) -> ObjectRef:
    _validate_identity(
        value.registry_id,
        value.registry_sha256,
        "attestation-issuer-registry",
        trusted_attestation_issuer_registry_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _ref(
        "attestation-issuer-registry",
        value.registry_id,
        "private-v1",
        value.registry_sha256,
    )


def attestation_issuance_proof_v1_ref(
    value: AttestationIssuanceProofV1,
) -> ObjectRef:
    _validate_identity(
        value.proof_id,
        value.proof_sha256,
        "attestation-issuance-proof",
        attestation_issuance_proof_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _ref(
        "attestation-issuance-proof",
        value.proof_id,
        "private-v1",
        value.proof_sha256,
    )


def attestation_issuance_bundle_v1_ref(
    value: AttestationIssuanceBundleV1,
) -> ObjectRef:
    _validate_identity(
        value.bundle_id,
        value.bundle_sha256,
        "attestation-issuance-bundle",
        attestation_issuance_bundle_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _ref(
        "attestation-issuance-bundle",
        value.bundle_id,
        "private-v1",
        value.bundle_sha256,
    )


def production_readiness_sc014_closure_v1_ref(
    value: ProductionReadinessSC014ClosureV1,
) -> ObjectRef:
    _validate_identity(
        value.closure_id,
        value.closure_sha256,
        "production-readiness-sc-014-closure",
        production_readiness_sc014_closure_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _ref(
        "production-readiness-sc-014-closure",
        value.closure_id,
        "private-v1",
        value.closure_sha256,
    )


def production_readiness_attestation_request_v1_ref(
    value: ProductionReadinessAttestationRequestV1,
) -> ObjectRef:
    _validate_identity(
        value.request_id,
        value.request_sha256,
        "production-readiness-attestation-request",
        production_readiness_attestation_request_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _ref(
        "production-readiness-attestation-request",
        value.request_id,
        "private-v1",
        value.request_sha256,
    )


def production_readiness_attestation_accepted_run_v1_ref(
    value: ProductionReadinessAttestationAcceptedRunV1,
) -> ObjectRef:
    _validate_identity(
        value.accepted_run_id,
        value.accepted_run_sha256,
        "production-attestation-accepted-run",
        production_readiness_attestation_accepted_run_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _ref(
        "production-attestation-accepted-run",
        value.accepted_run_id,
        "private-v1",
        value.accepted_run_sha256,
    )


def _request_refs(
    *,
    previous_result_ref: ObjectRef | None,
    previous_projection_ref: ObjectRef | None,
    label_quality_report: LabelQualityEvaluationReportV2,
    real_trace_stability_report: RealTraceStabilityReportV2,
    concurrency_experiment_report: ConcurrencyExperimentReportV2,
    production_readiness_review_report: ProductionReadinessReviewReportV2,
    version_set: ProductionReadinessVersionSetV2,
    expected_prerequisite: ProductionReadinessAttestationPrerequisiteV2,
    sc_014_closure: ProductionReadinessSC014ClosureV1,
    issuance_bundle: AttestationIssuanceBundleV1,
) -> tuple[ObjectRef, ...]:
    return _sorted_refs(
        (
            label_quality_report.to_ref(),
            real_trace_stability_report.to_ref(),
            concurrency_experiment_report.to_ref(),
            production_readiness_review_report.to_ref(),
            version_set.to_ref(),
            expected_prerequisite.to_ref(),
            sc_014_closure.to_ref(),
            issuance_bundle.to_ref(),
            *((previous_result_ref,) if previous_result_ref is not None else ()),
            *((previous_projection_ref,) if previous_projection_ref is not None else ()),
        )
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
    "ATTESTATION_ISSUER_VERIFICATION_METHOD",
    "AttestationIssuanceBundleV1",
    "AttestationIssuanceProofV1",
    "ProductionReadinessAttestationAcceptedRunV1",
    "ProductionReadinessAttestationRequestV1",
    "ProductionReadinessSC014ClosureV1",
    "RepositoryPendingAttestationEvidenceV1",
    "RepositoryPendingAttestationPredecessorV1",
    "TrustedAttestationIssuerRegistryV1",
    "TrustedAttestationIssuerV1",
    "attestation_issuance_bundle_v1_ref",
    "attestation_issuance_proof_v1_ref",
    "production_readiness_attestation_accepted_run_v1_ref",
    "production_readiness_attestation_request_v1_ref",
    "production_readiness_sc014_closure_v1_ref",
    "trusted_attestation_issuer_registry_v1_ref",
]
