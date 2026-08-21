from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.production_attestation_v2 import (
    ProductionReadinessAttestationOutcomeV2,
    ProductionReadinessAttestationProjectionV2,
    ProductionReadinessAttestationResultV2,
    ProductionReadinessAttestationStateV2,
    ProductionReadinessInvalidationRecordV2,
    ProductionReadinessInvalidationTriggerV2,
    ProductionReadinessVersionSetV2,
    validate_production_readiness_attestation_policy_v2_identity,
    validate_production_readiness_attestation_projection_v2_identity,
    validate_production_readiness_version_set_v2_identity,
)
from eval_factory.contracts.release import ProductionReadinessAttestation
from eval_factory.readiness.production_attestation_builder import (
    ProductionAttestationCompilation,
)

REPOSITORY_ATTESTATION_SERIES_ID = "production-attestation-series://repository/r8-08"


class ProductionAttestationError(RuntimeError):
    pass


class ProductionAttestationEvaluator:
    def evaluate(
        self,
        *,
        compilation: ProductionAttestationCompilation,
        audit: ContractAudit,
    ) -> tuple[
        ProductionReadinessAttestationResultV2,
        ProductionReadinessAttestation | None,
    ]:
        try:
            validate_production_readiness_attestation_policy_v2_identity(compilation.policy)
        except ValueError as exc:
            raise ProductionAttestationError("attestation policy is stale") from exc
        request = compilation.request
        if request is None:
            if (
                compilation.trusted_registry is not None
                or compilation.issuance_bundle is not None
                or compilation.approved_issuer_principals
            ):
                raise ProductionAttestationError("repository pending compilation carries issuance authority")
            result = ProductionReadinessAttestationResultV2.create(
                policy_ref=compilation.policy.to_ref(),
                version_set=compilation.version_set,
                prerequisite=compilation.prerequisite,
                evidence_class=compilation.evidence_class,
                attestation_series_id=REPOSITORY_ATTESTATION_SERIES_ID,
                attestation_version=1,
                previous_result_ref=None,
                projection=None,
                invalidation_record=None,
                private_request_closure_sha256=None,
                private_issuer_closure_sha256=None,
                private_attestation_closure_sha256=None,
                audit=audit,
            )
            return result, None
        if compilation.trusted_registry is None:
            raise ProductionAttestationError("full attestation compilation has no trusted registry")
        frozen: ProductionReadinessAttestation | None = None
        projection: ProductionReadinessAttestationProjectionV2 | None = None
        private_attestation_sha256: str | None = None
        if compilation.prerequisite.outcome is ProductionReadinessAttestationOutcomeV2.ISSUED:
            if (
                len(compilation.approved_issuer_principals) < compilation.policy.minimum_issuer_count
                or compilation.issuance_valid_until is None
            ):
                raise ProductionAttestationError("issuance compilation lacks independent authority")
            frozen = _create_frozen_attestation(
                compilation=compilation,
                audit=audit,
            )
            frozen_ref = production_readiness_attestation_v1_ref(frozen)
            private_attestation_sha256 = frozen_ref.object_sha256
            projection = ProductionReadinessAttestationProjectionV2.create_active(
                attestation_series_id=request.attestation_series_id,
                attestation_version=request.attestation_version,
                frozen_attestation_ref=frozen_ref,
                version_set_ref=compilation.version_set.to_ref(),
                prerequisite_ref=compilation.prerequisite.to_ref(),
                issuer_registry_ref=compilation.trusted_registry.to_ref(),
                issuer_count=len(compilation.approved_issuer_principals),
                valid_from=request.requested_valid_from,
                valid_until=compilation.issuance_valid_until,
                audit=audit,
            )
        result = ProductionReadinessAttestationResultV2.create(
            policy_ref=compilation.policy.to_ref(),
            version_set=compilation.version_set,
            prerequisite=compilation.prerequisite,
            evidence_class=compilation.evidence_class,
            attestation_series_id=request.attestation_series_id,
            attestation_version=request.attestation_version,
            previous_result_ref=request.previous_result_ref,
            projection=projection,
            invalidation_record=None,
            private_request_closure_sha256=(compilation.private_request_closure_sha256),
            private_issuer_closure_sha256=(compilation.private_issuer_closure_sha256),
            private_attestation_closure_sha256=private_attestation_sha256,
            audit=audit,
        )
        if len(result.canonical_json()) > compilation.policy.max_report_bytes:
            raise ProductionAttestationError("attestation result exceeds policy limit")
        return result, frozen

    def evaluate_currentness(
        self,
        *,
        projection: ProductionReadinessAttestationProjectionV2,
        prior_version_set: ProductionReadinessVersionSetV2,
        observed_version_set: ProductionReadinessVersionSetV2,
        observed_issuer_registry_ref: ObjectRef,
        observed_at: datetime,
        evidence_superseded: bool = False,
        issuer_revoked_or_expired: bool = False,
        explicit_revocation_kind: Literal["INCIDENT", "RELEASE"] | None = None,
        explicit_revocation_ref: ObjectRef | None = None,
        audit: ContractAudit,
    ) -> tuple[
        ProductionReadinessInvalidationRecordV2 | None,
        ProductionReadinessAttestationProjectionV2,
    ]:
        try:
            projection = ProductionReadinessAttestationProjectionV2.model_validate_json(
                projection.canonical_json()
            )
            prior_version_set = ProductionReadinessVersionSetV2.model_validate_json(
                prior_version_set.canonical_json()
            )
            observed_version_set = ProductionReadinessVersionSetV2.model_validate_json(
                observed_version_set.canonical_json()
            )
            validate_production_readiness_attestation_projection_v2_identity(projection)
            validate_production_readiness_version_set_v2_identity(prior_version_set)
            validate_production_readiness_version_set_v2_identity(observed_version_set)
        except ValueError as exc:
            raise ProductionAttestationError("attestation currentness authority is invalid") from exc
        if projection.state is not ProductionReadinessAttestationStateV2.ACTIVE:
            raise ProductionAttestationError("invalidated or expired attestation cannot reactivate")
        if projection.version_set_ref != prior_version_set.to_ref():
            raise ProductionAttestationError("attestation prior version set differs from projection")
        if (
            observed_issuer_registry_ref.object_type != "attestation-issuer-registry"
            or observed_issuer_registry_ref.object_version != "private-v1"
        ):
            raise ProductionAttestationError("observed issuer registry ref has invalid type")
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ProductionAttestationError("attestation currentness time must be timezone-aware")
        if observed_at < projection.valid_from:
            raise ProductionAttestationError("attestation is outside its validity window")
        triggers: set[ProductionReadinessInvalidationTriggerV2] = set()
        if prior_version_set.system_version != observed_version_set.system_version:
            triggers.add(ProductionReadinessInvalidationTriggerV2.SYSTEM_VERSION_CHANGED)
        if prior_version_set.base_contract_manifest_ref != observed_version_set.base_contract_manifest_ref:
            triggers.add(ProductionReadinessInvalidationTriggerV2.BASE_CONTRACT_MANIFEST_CHANGED)
        if (
            prior_version_set.overlay_contract_manifest_ref
            != observed_version_set.overlay_contract_manifest_ref
        ):
            triggers.add(ProductionReadinessInvalidationTriggerV2.OVERLAY_CONTRACT_MANIFEST_CHANGED)
        if prior_version_set.schema_manifest_refs != observed_version_set.schema_manifest_refs:
            triggers.add(ProductionReadinessInvalidationTriggerV2.SCHEMA_SET_CHANGED)
        if (
            prior_version_set.policy_refs != observed_version_set.policy_refs
            or prior_version_set.release_profile_decision_ref
            != observed_version_set.release_profile_decision_ref
            or prior_version_set.resource_policy_approval_ref
            != observed_version_set.resource_policy_approval_ref
            or prior_version_set.lifecycle_approval_ref != observed_version_set.lifecycle_approval_ref
            or prior_version_set.incident_route_ref != observed_version_set.incident_route_ref
        ):
            triggers.add(ProductionReadinessInvalidationTriggerV2.POLICY_SET_CHANGED)
        if evidence_superseded:
            triggers.add(ProductionReadinessInvalidationTriggerV2.EVIDENCE_SUPERSEDED)
        if projection.issuer_registry_ref != observed_issuer_registry_ref:
            triggers.add(ProductionReadinessInvalidationTriggerV2.ISSUER_REGISTRY_CHANGED)
        if issuer_revoked_or_expired:
            triggers.add(ProductionReadinessInvalidationTriggerV2.ISSUER_REVOKED_OR_EXPIRED)
        if observed_at >= projection.valid_until:
            triggers.add(ProductionReadinessInvalidationTriggerV2.ATTESTATION_EXPIRED)
        if explicit_revocation_kind is not None:
            if explicit_revocation_ref is None:
                raise ProductionAttestationError("explicit revocation requires an immutable ref")
            triggers.add(
                ProductionReadinessInvalidationTriggerV2.EXPLICIT_INCIDENT_REVOCATION
                if explicit_revocation_kind == "INCIDENT"
                else ProductionReadinessInvalidationTriggerV2.EXPLICIT_RELEASE_REVOCATION
            )
        elif explicit_revocation_ref is not None:
            raise ProductionAttestationError("explicit revocation ref has no closed revocation kind")
        if not triggers:
            return None, projection
        ordered = tuple(item for item in ProductionReadinessInvalidationTriggerV2 if item in triggers)
        record = ProductionReadinessInvalidationRecordV2.create(
            prior_projection_ref=projection.to_ref(),
            prior_version_set_ref=prior_version_set.to_ref(),
            observed_version_set_ref=observed_version_set.to_ref(),
            triggers=ordered,
            explicit_revocation_ref=explicit_revocation_ref,
            observed_at=observed_at,
            audit=audit,
        )
        state = (
            ProductionReadinessAttestationStateV2.EXPIRED
            if set(ordered) == {ProductionReadinessInvalidationTriggerV2.ATTESTATION_EXPIRED}
            else ProductionReadinessAttestationStateV2.INVALIDATED
        )
        successor = ProductionReadinessAttestationProjectionV2.create_successor(
            prior=projection,
            state=state,
            invalidation_record=record,
            audit=audit,
        )
        return record, successor


def production_readiness_attestation_v1_carried_sha256(
    value: ProductionReadinessAttestation,
) -> str:
    payload = value.model_dump(
        mode="python",
        exclude={"attestation_id", "attestation_sha256"},
        exclude_none=False,
    )
    return hashlib.sha256(
        json.dumps(
            payload,
            default=lambda item: item.isoformat() if isinstance(item, datetime) else item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def validate_production_readiness_attestation_v1_identity(
    value: ProductionReadinessAttestation,
) -> None:
    digest = production_readiness_attestation_v1_carried_sha256(value)
    if (
        value.attestation_sha256 != digest
        or value.attestation_id != f"production-readiness-attestation://sha256/{digest}"
    ):
        raise ValueError("production readiness attestation v1 identity is stale")


def production_readiness_attestation_v1_ref(
    value: ProductionReadinessAttestation,
) -> ObjectRef:
    validate_production_readiness_attestation_v1_identity(value)
    return ObjectRef(
        object_type="production-readiness-attestation",
        object_id=value.attestation_id,
        object_version="v1",
        object_sha256=value.attestation_sha256,
    )


def _create_frozen_attestation(
    *,
    compilation: ProductionAttestationCompilation,
    audit: ContractAudit,
) -> ProductionReadinessAttestation:
    del audit
    request = compilation.request
    if request is None or compilation.issuance_valid_until is None:
        raise ProductionAttestationError("frozen attestation request is missing")
    version_set = compilation.version_set
    schema_refs = tuple(
        sorted(
            (
                version_set.base_contract_manifest_ref,
                version_set.overlay_contract_manifest_ref,
                *version_set.schema_manifest_refs,
            ),
            key=lambda value: (
                value.object_type,
                value.object_id,
                value.object_version,
                value.object_sha256,
            ),
        )
    )
    value = ProductionReadinessAttestation(
        attestation_id="production-readiness-attestation://pending",
        system_version=version_set.system_version,
        contract_manifest_ref=version_set.overlay_contract_manifest_ref,
        policy_refs=version_set.policy_refs,
        schema_refs=schema_refs,
        statistical_evidence_ref=request.label_quality_report.to_ref(),
        safety_evidence_ref=request.production_readiness_review_report.to_ref(),
        privacy_evidence_ref=request.production_readiness_review_report.to_ref(),
        stability_evidence_ref=request.real_trace_stability_report.to_ref(),
        operations_evidence_ref=request.production_readiness_review_report.to_ref(),
        approved_by=compilation.approved_issuer_principals,
        valid_from=request.requested_valid_from,
        valid_until=compilation.issuance_valid_until,
        attestation_sha256="0" * 64,
    )
    digest = production_readiness_attestation_v1_carried_sha256(value)
    return value.model_copy(
        update={
            "attestation_id": (f"production-readiness-attestation://sha256/{digest}"),
            "attestation_sha256": digest,
        }
    )


__all__ = [
    "REPOSITORY_ATTESTATION_SERIES_ID",
    "ProductionAttestationError",
    "ProductionAttestationEvaluator",
    "production_readiness_attestation_v1_carried_sha256",
    "production_readiness_attestation_v1_ref",
    "validate_production_readiness_attestation_v1_identity",
]
