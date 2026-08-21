from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.canary_regression_v2 import (
    CanaryRegressionReportV2,
    validate_canary_regression_report_v2_identity,
)
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
from eval_factory.contracts.production_readiness_review_v2 import (
    PRODUCTION_READINESS_DOMAIN_ORDER,
    OperationsSLOAssessmentV2,
    OperationsSLOPolicyV2,
    ProductionReadinessApprovalDecisionV2,
    ProductionReadinessReviewDomainV2,
    operations_slo_assessment_v2_ref,
    operations_slo_policy_v2_ref,
    validate_operations_slo_assessment_v2_identity,
    validate_operations_slo_policy_v2_identity,
)
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityReportV2,
    validate_real_trace_stability_report_v2_identity,
)

PRODUCTION_APPROVAL_VERIFICATION_METHOD: Literal["DETACHED_REGISTRY_PROOF_V1"] = "DETACHED_REGISTRY_PROOF_V1"


class TrustedProductionApprovalAuthorityV1(ContractModelV2):
    schema_version: Literal["eval-factory/trusted-production-approval-authority/private-v1"] = (
        "eval-factory/trusted-production-approval-authority/private-v1"
    )
    authority_principal: Identifier
    authority_role: Identifier
    approved_domains: tuple[ProductionReadinessReviewDomainV2, ...] = Field(
        min_length=1,
        max_length=4,
    )
    organization_policy_id: Identifier
    organization_policy_version: str = Field(min_length=1, max_length=128)
    verification_method: Literal["DETACHED_REGISTRY_PROOF_V1"] = PRODUCTION_APPROVAL_VERIFICATION_METHOD
    valid_from: datetime
    valid_until: datetime
    registry_evidence_sha256: Sha256

    @field_validator("approved_domains")
    @classmethod
    def validate_domain_order(
        cls,
        value: tuple[ProductionReadinessReviewDomainV2, ...],
    ) -> tuple[ProductionReadinessReviewDomainV2, ...]:
        expected = tuple(sorted(set(value), key=_domain_index))
        if value != expected:
            raise ValueError("authority domains must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_authority(self) -> Self:
        _require_aware(self.valid_from, "authority valid_from")
        _require_aware(self.valid_until, "authority valid_until")
        if self.valid_until <= self.valid_from:
            raise ValueError("authority validity window is invalid")
        return self


class TrustedProductionApprovalRegistryV1(ContractModelV2):
    schema_version: Literal["eval-factory/trusted-production-approval-registry/private-v1"] = (
        "eval-factory/trusted-production-approval-registry/private-v1"
    )
    registry_id: Identifier
    registry_version: str = Field(min_length=1, max_length=128)
    project_owner_principal: Identifier
    authorities: tuple[TrustedProductionApprovalAuthorityV1, ...] = Field(
        min_length=1,
        max_length=1_000,
    )
    registry_proof_commitment_sha256: Sha256
    valid_from: datetime
    valid_until: datetime
    registry_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_registry(self) -> Self:
        principals = tuple(value.authority_principal for value in self.authorities)
        if principals != tuple(sorted(principals)) or len(principals) != len(set(principals)):
            raise ValueError("registry authorities must be sorted and unique")
        _require_aware(self.valid_from, "registry valid_from")
        _require_aware(self.valid_until, "registry valid_until")
        if self.valid_until <= self.valid_from:
            raise ValueError("registry validity window is invalid")
        _require_audit(self.audit, (), "trusted production approval registry")
        _validate_identity(
            self.registry_id,
            self.registry_sha256,
            "production-approval-authority-registry",
            trusted_production_approval_registry_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        registry_version: str,
        project_owner_principal: str,
        authorities: tuple[TrustedProductionApprovalAuthorityV1, ...],
        registry_proof_commitment_sha256: str,
        valid_from: datetime,
        valid_until: datetime,
        audit: ContractAudit,
    ) -> TrustedProductionApprovalRegistryV1:
        ordered = tuple(sorted(authorities, key=lambda value: value.authority_principal))
        value = cls(
            registry_id="production-approval-authority-registry://pending",
            registry_version=registry_version,
            project_owner_principal=project_owner_principal,
            authorities=ordered,
            registry_proof_commitment_sha256=(registry_proof_commitment_sha256),
            valid_from=valid_from,
            valid_until=valid_until,
            registry_sha256="0" * 64,
            audit=_safe_audit(audit, ()),
        )
        return _finalize(
            value,
            "registry_id",
            "registry_sha256",
            "production-approval-authority-registry",
            trusted_production_approval_registry_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return trusted_production_approval_registry_v1_ref(self)


class ProductionApprovalProofV1(ContractModelV2):
    schema_version: Literal["eval-factory/production-approval-proof/private-v1"] = (
        "eval-factory/production-approval-proof/private-v1"
    )
    proof_id: Identifier
    authority_principal: Identifier
    authority_role: Identifier
    organization_policy_id: Identifier
    organization_policy_version: str = Field(min_length=1, max_length=128)
    verification_method: Literal["DETACHED_REGISTRY_PROOF_V1"] = PRODUCTION_APPROVAL_VERIFICATION_METHOD
    domain: ProductionReadinessReviewDomainV2
    decision: ProductionReadinessApprovalDecisionV2
    review_series_id: Identifier
    review_version: int = Field(ge=1, le=1_000_000)
    subject_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=1_000)
    evidence_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=1_000)
    policy_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=1_000)
    proof_value: str = Field(min_length=16, max_length=16_384)
    proof_value_sha256: Sha256
    issued_at: datetime
    valid_from: datetime
    valid_until: datetime
    proof_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_proof(self) -> Self:
        for name, refs in (
            ("approval proof subject refs", self.subject_refs),
            ("approval proof evidence refs", self.evidence_refs),
            ("approval proof policy refs", self.policy_refs),
        ):
            _require_sorted_unique_refs(refs, name)
        if hashlib.sha256(self.proof_value.encode()).hexdigest() != (self.proof_value_sha256):
            raise ValueError("approval proof value hash is stale")
        for value, name in (
            (self.issued_at, "proof issued_at"),
            (self.valid_from, "proof valid_from"),
            (self.valid_until, "proof valid_until"),
        ):
            _require_aware(value, name)
        if self.valid_until <= self.valid_from or self.issued_at > self.valid_from:
            raise ValueError("approval proof validity window is invalid")
        refs = (*self.subject_refs, *self.evidence_refs, *self.policy_refs)
        _require_audit(self.audit, refs, "production approval proof")
        _validate_identity(
            self.proof_id,
            self.proof_sha256,
            "production-approval-proof",
            production_approval_proof_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        authority_principal: str,
        authority_role: str,
        organization_policy_id: str,
        organization_policy_version: str,
        domain: ProductionReadinessReviewDomainV2,
        decision: ProductionReadinessApprovalDecisionV2,
        review_series_id: str,
        review_version: int,
        subject_refs: tuple[ObjectRef, ...],
        evidence_refs: tuple[ObjectRef, ...],
        policy_refs: tuple[ObjectRef, ...],
        proof_value: str,
        issued_at: datetime,
        valid_from: datetime,
        valid_until: datetime,
        audit: ContractAudit,
    ) -> ProductionApprovalProofV1:
        subjects = _sorted_refs(subject_refs)
        evidence = _sorted_refs(evidence_refs)
        policies = _sorted_refs(policy_refs)
        value = cls(
            proof_id="production-approval-proof://pending",
            authority_principal=authority_principal,
            authority_role=authority_role,
            organization_policy_id=organization_policy_id,
            organization_policy_version=organization_policy_version,
            domain=domain,
            decision=decision,
            review_series_id=review_series_id,
            review_version=review_version,
            subject_refs=subjects,
            evidence_refs=evidence,
            policy_refs=policies,
            proof_value=proof_value,
            proof_value_sha256=hashlib.sha256(proof_value.encode()).hexdigest(),
            issued_at=issued_at,
            valid_from=valid_from,
            valid_until=valid_until,
            proof_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                (*subjects, *evidence, *policies),
            ),
        )
        return _finalize(
            value,
            "proof_id",
            "proof_sha256",
            "production-approval-proof",
            production_approval_proof_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return production_approval_proof_v1_ref(self)


class ProductionApprovalBundleV1(ContractModelV2):
    schema_version: Literal["eval-factory/production-approval-bundle/private-v1"] = (
        "eval-factory/production-approval-bundle/private-v1"
    )
    bundle_id: Identifier
    review_series_id: Identifier
    review_version: int = Field(ge=1, le=1_000_000)
    proofs: tuple[ProductionApprovalProofV1, ...] = Field(
        min_length=1,
        max_length=4,
    )
    bundle_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        domains = tuple(value.domain for value in self.proofs)
        if domains != tuple(sorted(domains, key=_domain_index)) or len(domains) != len(set(domains)):
            raise ValueError("approval bundle domains must be sorted and unique")
        for proof in self.proofs:
            if proof.review_series_id != self.review_series_id or proof.review_version != self.review_version:
                raise ValueError("approval proof review version differs from bundle")
        refs = tuple(value.to_ref() for value in self.proofs)
        _require_audit(self.audit, refs, "production approval bundle")
        _validate_identity(
            self.bundle_id,
            self.bundle_sha256,
            "production-approval-bundle",
            production_approval_bundle_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        review_series_id: str,
        review_version: int,
        proofs: tuple[ProductionApprovalProofV1, ...],
        audit: ContractAudit,
    ) -> ProductionApprovalBundleV1:
        ordered = tuple(sorted(proofs, key=lambda value: _domain_index(value.domain)))
        refs = tuple(value.to_ref() for value in ordered)
        value = cls(
            bundle_id="production-approval-bundle://pending",
            review_series_id=review_series_id,
            review_version=review_version,
            proofs=ordered,
            bundle_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "bundle_id",
            "bundle_sha256",
            "production-approval-bundle",
            production_approval_bundle_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return production_approval_bundle_v1_ref(self)


class ProductionReadinessReviewRequestV1(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-review-request/private-v1"] = (
        "eval-factory/production-readiness-review-request/private-v1"
    )
    request_id: Identifier
    review_series_id: Identifier
    review_version: int = Field(ge=1, le=1_000_000)
    previous_report_ref: ObjectRef | None = None
    label_quality_report: LabelQualityEvaluationReportV2
    canary_regression_report: CanaryRegressionReportV2
    real_trace_stability_report: RealTraceStabilityReportV2
    concurrency_experiment_report: ConcurrencyExperimentReportV2
    operations_slo_policy: OperationsSLOPolicyV2 | None = None
    operations_slo_assessment: OperationsSLOAssessmentV2 | None = None
    approval_bundle: ProductionApprovalBundleV1 | None = None
    evaluated_at: datetime
    request_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        try:
            validate_label_quality_evaluation_report_v2_identity(self.label_quality_report)
            validate_canary_regression_report_v2_identity(self.canary_regression_report)
            validate_real_trace_stability_report_v2_identity(self.real_trace_stability_report)
            validate_concurrency_experiment_report_v2_identity(self.concurrency_experiment_report)
        except ValueError as exc:
            raise ValueError("review request predecessor identity is stale") from exc
        if self.review_version == 1:
            if self.previous_report_ref is not None:
                raise ValueError("review request version 1 cannot have predecessor")
        else:
            if self.previous_report_ref is None:
                raise ValueError("later review request requires previous report")
            _require_ref(
                self.previous_report_ref,
                "production-readiness-review-report",
                "v2",
                "previous_report_ref",
            )
        _require_aware(self.evaluated_at, "review evaluated_at")
        validate_production_readiness_review_request_v1_slo_closure(self)
        if self.approval_bundle is not None and (
            self.approval_bundle.review_series_id != self.review_series_id
            or self.approval_bundle.review_version != self.review_version
        ):
            raise ValueError("approval bundle differs from review request")
        refs = self.refs
        _require_audit(self.audit, refs, "production readiness review request")
        _validate_identity(
            self.request_id,
            self.request_sha256,
            "production-readiness-review-request",
            production_readiness_review_request_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        review_series_id: str,
        review_version: int,
        previous_report_ref: ObjectRef | None,
        label_quality_report: LabelQualityEvaluationReportV2,
        canary_regression_report: CanaryRegressionReportV2,
        real_trace_stability_report: RealTraceStabilityReportV2,
        concurrency_experiment_report: ConcurrencyExperimentReportV2,
        operations_slo_policy: OperationsSLOPolicyV2 | None,
        operations_slo_assessment: OperationsSLOAssessmentV2 | None,
        approval_bundle: ProductionApprovalBundleV1 | None,
        evaluated_at: datetime,
        audit: ContractAudit,
    ) -> ProductionReadinessReviewRequestV1:
        refs = _request_refs(
            previous_report_ref=previous_report_ref,
            label_quality_report=label_quality_report,
            canary_regression_report=canary_regression_report,
            real_trace_stability_report=real_trace_stability_report,
            concurrency_experiment_report=concurrency_experiment_report,
            operations_slo_policy=operations_slo_policy,
            operations_slo_assessment=operations_slo_assessment,
            approval_bundle=approval_bundle,
        )
        value = cls(
            request_id="production-readiness-review-request://pending",
            review_series_id=review_series_id,
            review_version=review_version,
            previous_report_ref=previous_report_ref,
            label_quality_report=label_quality_report,
            canary_regression_report=canary_regression_report,
            real_trace_stability_report=real_trace_stability_report,
            concurrency_experiment_report=concurrency_experiment_report,
            operations_slo_policy=operations_slo_policy,
            operations_slo_assessment=operations_slo_assessment,
            approval_bundle=approval_bundle,
            evaluated_at=evaluated_at,
            request_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "request_id",
            "request_sha256",
            "production-readiness-review-request",
            production_readiness_review_request_v1_carried_sha256(value),
        )

    @property
    def refs(self) -> tuple[ObjectRef, ...]:
        return _request_refs(
            previous_report_ref=self.previous_report_ref,
            label_quality_report=self.label_quality_report,
            canary_regression_report=self.canary_regression_report,
            real_trace_stability_report=self.real_trace_stability_report,
            concurrency_experiment_report=self.concurrency_experiment_report,
            operations_slo_policy=self.operations_slo_policy,
            operations_slo_assessment=self.operations_slo_assessment,
            approval_bundle=self.approval_bundle,
        )

    def to_ref(self) -> ObjectRef:
        return production_readiness_review_request_v1_ref(self)


def validate_production_readiness_review_request_v1_slo_closure(
    value: ProductionReadinessReviewRequestV1,
) -> None:
    policy = value.operations_slo_policy
    assessment = value.operations_slo_assessment
    if (policy is None) != (assessment is None):
        raise ValueError("operations SLO request closure must be all-or-none")
    if policy is None or assessment is None:
        return
    validate_operations_slo_policy_v2_identity(policy)
    validate_operations_slo_assessment_v2_identity(assessment)
    if assessment.policy_ref != policy.to_ref():
        raise ValueError("operations SLO assessment differs from policy")
    if not policy.valid_from <= value.evaluated_at < policy.valid_until:
        raise ValueError("operations SLO policy is not current")
    for metric in assessment.metrics:
        if (
            metric.threshold_value != policy.threshold_for(metric.metric)
            or metric.required_sample_count != policy.minimum_sample_count
        ):
            raise ValueError("operations SLO assessment threshold differs from policy")


class RepositoryPendingPredecessorV1(ContractModelV2):
    schema_version: Literal["eval-factory/repository-pending-readiness-predecessor/private-v1"] = (
        "eval-factory/repository-pending-readiness-predecessor/private-v1"
    )
    stage: Literal["R8_02", "R8_03", "R8_04", "R8_06"]
    report_sha256: Sha256
    outcome: str = Field(min_length=1, max_length=128)
    claim_scope: str = Field(min_length=1, max_length=128)
    satisfies_stage_gate: bool


class RepositoryPendingReviewEvidenceV1(ContractModelV2):
    schema_version: Literal["eval-factory/repository-pending-readiness-review-evidence/v1"] = (
        "eval-factory/repository-pending-readiness-review-evidence/v1"
    )
    evidence_class: Literal["REPOSITORY_PENDING_ONLY"]
    predecessors: tuple[RepositoryPendingPredecessorV1, ...] = Field(
        min_length=4,
        max_length=4,
    )
    missing_approval_domains: tuple[ProductionReadinessReviewDomainV2, ...] = Field(
        min_length=4, max_length=4
    )
    organization_approval_authority_present: Literal[False]
    operations_slo_authority_present: Literal[False]
    authorizes_attestation: Literal[False]
    authorizes_production_release: Literal[False]

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if tuple(value.stage for value in self.predecessors) != (
            "R8_02",
            "R8_03",
            "R8_04",
            "R8_06",
        ):
            raise ValueError("repository predecessor inventory is not canonical")
        if self.missing_approval_domains != PRODUCTION_READINESS_DOMAIN_ORDER:
            raise ValueError("repository missing-domain inventory is not canonical")
        return self


class ProductionReadinessAcceptedRunV1(ContractModelV2):
    schema_version: Literal["eval-factory/production-readiness-accepted-run/private-v1"] = (
        "eval-factory/production-readiness-accepted-run/private-v1"
    )
    accepted_run_id: Identifier
    acceptance_key: Identifier
    request_sha256: Sha256
    report_ref: ObjectRef
    registry_ref: ObjectRef | None = None
    request_ref: ObjectRef | None = None
    approval_bundle_ref: ObjectRef | None = None
    accepted_run_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        _require_ref(
            self.report_ref,
            "production-readiness-review-report",
            "v2",
            "report_ref",
        )
        if (self.registry_ref is None) != (self.request_ref is None):
            raise ValueError("accepted run full closure must be all-or-none")
        if self.registry_ref is not None:
            _require_ref(
                self.registry_ref,
                "production-approval-authority-registry",
                "private-v1",
                "registry_ref",
            )
            if self.request_ref is None:
                raise ValueError("accepted run request ref is missing")
            _require_ref(
                self.request_ref,
                "production-readiness-review-request",
                "private-v1",
                "request_ref",
            )
        elif self.approval_bundle_ref is not None:
            raise ValueError("pending accepted run cannot carry approval bundle")
        if self.approval_bundle_ref is not None:
            _require_ref(
                self.approval_bundle_ref,
                "production-approval-bundle",
                "private-v1",
                "approval_bundle_ref",
            )
        refs = (
            self.report_ref,
            *((self.registry_ref,) if self.registry_ref is not None else ()),
            *((self.request_ref,) if self.request_ref is not None else ()),
            *((self.approval_bundle_ref,) if self.approval_bundle_ref is not None else ()),
        )
        _require_audit(self.audit, refs, "production readiness accepted run")
        _validate_identity(
            self.accepted_run_id,
            self.accepted_run_sha256,
            "production-readiness-accepted-run",
            production_readiness_accepted_run_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        acceptance_key: str,
        request_sha256: str,
        report_ref: ObjectRef,
        registry_ref: ObjectRef | None,
        request_ref: ObjectRef | None,
        approval_bundle_ref: ObjectRef | None,
        audit: ContractAudit,
    ) -> ProductionReadinessAcceptedRunV1:
        refs = (
            report_ref,
            *((registry_ref,) if registry_ref is not None else ()),
            *((request_ref,) if request_ref is not None else ()),
            *((approval_bundle_ref,) if approval_bundle_ref is not None else ()),
        )
        value = cls(
            accepted_run_id="production-readiness-accepted-run://pending",
            acceptance_key=acceptance_key,
            request_sha256=request_sha256,
            report_ref=report_ref,
            registry_ref=registry_ref,
            request_ref=request_ref,
            approval_bundle_ref=approval_bundle_ref,
            accepted_run_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "accepted_run_id",
            "accepted_run_sha256",
            "production-readiness-accepted-run",
            production_readiness_accepted_run_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return production_readiness_accepted_run_v1_ref(self)


def trusted_production_approval_registry_v1_carried_sha256(
    value: TrustedProductionApprovalRegistryV1,
) -> str:
    return _carried(value, {"registry_id", "registry_sha256", "audit"})


def production_approval_proof_v1_carried_sha256(
    value: ProductionApprovalProofV1,
) -> str:
    return _carried(value, {"proof_id", "proof_sha256", "audit"})


def production_approval_bundle_v1_carried_sha256(
    value: ProductionApprovalBundleV1,
) -> str:
    return _carried(value, {"bundle_id", "bundle_sha256", "audit"})


def production_readiness_review_request_v1_carried_sha256(
    value: ProductionReadinessReviewRequestV1,
) -> str:
    return _carried(value, {"request_id", "request_sha256", "audit"})


def production_readiness_accepted_run_v1_carried_sha256(
    value: ProductionReadinessAcceptedRunV1,
) -> str:
    return _carried(value, {"accepted_run_id", "accepted_run_sha256", "audit"})


def trusted_production_approval_registry_v1_ref(
    value: TrustedProductionApprovalRegistryV1,
) -> ObjectRef:
    _validate_identity(
        value.registry_id,
        value.registry_sha256,
        "production-approval-authority-registry",
        trusted_production_approval_registry_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _ref(
        "production-approval-authority-registry",
        value.registry_id,
        "private-v1",
        value.registry_sha256,
    )


def production_approval_proof_v1_ref(
    value: ProductionApprovalProofV1,
) -> ObjectRef:
    _validate_identity(
        value.proof_id,
        value.proof_sha256,
        "production-approval-proof",
        production_approval_proof_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _ref(
        "production-approval-proof",
        value.proof_id,
        "private-v1",
        value.proof_sha256,
    )


def production_approval_bundle_v1_ref(
    value: ProductionApprovalBundleV1,
) -> ObjectRef:
    _validate_identity(
        value.bundle_id,
        value.bundle_sha256,
        "production-approval-bundle",
        production_approval_bundle_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _ref(
        "production-approval-bundle",
        value.bundle_id,
        "private-v1",
        value.bundle_sha256,
    )


def production_readiness_review_request_v1_ref(
    value: ProductionReadinessReviewRequestV1,
) -> ObjectRef:
    _validate_identity(
        value.request_id,
        value.request_sha256,
        "production-readiness-review-request",
        production_readiness_review_request_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _ref(
        "production-readiness-review-request",
        value.request_id,
        "private-v1",
        value.request_sha256,
    )


def production_readiness_accepted_run_v1_ref(
    value: ProductionReadinessAcceptedRunV1,
) -> ObjectRef:
    _validate_identity(
        value.accepted_run_id,
        value.accepted_run_sha256,
        "production-readiness-accepted-run",
        production_readiness_accepted_run_v1_carried_sha256(value),
        allow_pending=False,
    )
    return _ref(
        "production-readiness-accepted-run",
        value.accepted_run_id,
        "private-v1",
        value.accepted_run_sha256,
    )


def _request_refs(
    *,
    previous_report_ref: ObjectRef | None,
    label_quality_report: LabelQualityEvaluationReportV2,
    canary_regression_report: CanaryRegressionReportV2,
    real_trace_stability_report: RealTraceStabilityReportV2,
    concurrency_experiment_report: ConcurrencyExperimentReportV2,
    operations_slo_policy: OperationsSLOPolicyV2 | None,
    operations_slo_assessment: OperationsSLOAssessmentV2 | None,
    approval_bundle: ProductionApprovalBundleV1 | None,
) -> tuple[ObjectRef, ...]:
    return _sorted_refs(
        (
            label_quality_report.to_ref(),
            canary_regression_report.to_ref(),
            real_trace_stability_report.to_ref(),
            concurrency_experiment_report.to_ref(),
            *((previous_report_ref,) if previous_report_ref is not None else ()),
            *(
                (operations_slo_policy_v2_ref(operations_slo_policy),)
                if operations_slo_policy is not None
                else ()
            ),
            *(
                (operations_slo_assessment_v2_ref(operations_slo_assessment),)
                if operations_slo_assessment is not None
                else ()
            ),
            *((production_approval_bundle_v1_ref(approval_bundle),) if approval_bundle is not None else ()),
        )
    )


def _domain_index(value: ProductionReadinessReviewDomainV2) -> int:
    return PRODUCTION_READINESS_DOMAIN_ORDER.index(value)


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
