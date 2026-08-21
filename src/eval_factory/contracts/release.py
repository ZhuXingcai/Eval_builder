from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from eval_factory.contracts.core import ContractAudit, ContractModel, Identifier, ObjectRef, Sha256


class ReleaseChannel(StrEnum):
    CANARY = "CANARY"
    INTERNAL_REVIEW = "INTERNAL_REVIEW"
    PRODUCTION = "PRODUCTION"


class ReleaseState(StrEnum):
    CANDIDATE = "CANDIDATE"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    REJECTED = "REJECTED"
    APPROVED = "APPROVED"
    RELEASED = "RELEASED"
    REVOKED = "REVOKED"


class ReleaseAction(StrEnum):
    SUBMIT_REVIEW = "SUBMIT_REVIEW"
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    PUBLISH = "PUBLISH"
    REVOKE = "REVOKE"


class EvaluationItemComponents(ContractModel):
    schema_version: Literal["eval-factory/evaluation-item-components/v1"] = (
        "eval-factory/evaluation-item-components/v1"
    )
    query_spec_ref: ObjectRef
    environment_spec_ref: ObjectRef
    rubric_set_ref: ObjectRef
    evaluator_spec_ref: ObjectRef
    reference_policy_ref: ObjectRef
    tool_policy_ref: ObjectRef
    provenance_manifest_ref: ObjectRef
    quality_report_ref: ObjectRef


class ProductionReadinessAttestation(ContractModel):
    schema_version: Literal["eval-factory/production-readiness-attestation/v1"] = (
        "eval-factory/production-readiness-attestation/v1"
    )
    attestation_id: Identifier
    system_version: str = Field(min_length=1, max_length=128)
    contract_manifest_ref: ObjectRef
    policy_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    schema_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    statistical_evidence_ref: ObjectRef
    safety_evidence_ref: ObjectRef
    privacy_evidence_ref: ObjectRef
    stability_evidence_ref: ObjectRef
    operations_evidence_ref: ObjectRef
    approved_by: tuple[Identifier, ...] = Field(min_length=2)
    valid_from: datetime
    valid_until: datetime
    attestation_sha256: Sha256

    @model_validator(mode="after")
    def validate_window(self) -> ProductionReadinessAttestation:
        if self.valid_until <= self.valid_from:
            raise ValueError("attestation expiry must follow activation")
        return self


class ReleaseDecision(ContractModel):
    schema_version: Literal["eval-factory/release-decision/v1"] = "eval-factory/release-decision/v1"
    release_decision_id: Identifier
    chain_id: Identifier
    previous_decision_ref: ObjectRef | None
    item_id: Identifier
    item_version: str = Field(min_length=1, max_length=64)
    components: EvaluationItemComponents
    release_subject_sha256: Sha256
    package_manifest_ref: ObjectRef
    package_sha256: Sha256
    batch_quality_report_ref: ObjectRef | None
    human_review_record_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    review_quorum_refs: tuple[ObjectRef, ...] = ()
    review_policy_versions: tuple[str, ...] = Field(min_length=1)
    channel: ReleaseChannel
    registry: Identifier
    export_profile: Literal["LH", "GENERIC"]
    export_profile_version: str = Field(min_length=1, max_length=128)
    production_attestation_ref: ObjectRef | None
    action: ReleaseAction
    state: ReleaseState
    idempotency_key: Identifier
    actor: Identifier
    decided_at: datetime
    decision_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_release(self) -> ReleaseDecision:
        if self.channel is ReleaseChannel.PRODUCTION:
            if self.production_attestation_ref is None:
                raise ValueError("production decision requires readiness attestation")
        elif self.production_attestation_ref is not None:
            raise ValueError("non-production decision must not bind production attestation")
        action_states = {
            ReleaseAction.SUBMIT_REVIEW: ReleaseState.NEEDS_REVIEW,
            ReleaseAction.APPROVE: ReleaseState.APPROVED,
            ReleaseAction.REJECT: ReleaseState.REJECTED,
            ReleaseAction.PUBLISH: ReleaseState.RELEASED,
            ReleaseAction.REVOKE: ReleaseState.REVOKED,
        }
        if self.state is not action_states[self.action]:
            raise ValueError(f"{self.action} action must create {action_states[self.action]} state")
        return self


class EvaluationItem(ContractModel):
    schema_version: Literal["eval-factory/evaluation-item/v1"] = "eval-factory/evaluation-item/v1"
    evaluation_item_id: Identifier
    item_version: str = Field(min_length=1, max_length=64)
    source_trace_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    label_decision_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    task_draft_ref: ObjectRef
    attachment_reconstruction_result_ref: ObjectRef
    components: EvaluationItemComponents
    release_decision_ref: ObjectRef
    item_sha256: Sha256
    audit: ContractAudit


class CompatibilityImpact(StrEnum):
    NO_EFFECT = "NO_EFFECT"
    REVALIDATE = "REVALIDATE"
    INVALIDATE = "INVALIDATE"


class CompatibilityDeclaration(ContractModel):
    schema_version: Literal["eval-factory/compatibility-declaration/v1"] = (
        "eval-factory/compatibility-declaration/v1"
    )
    declaration_id: Identifier
    changed_contract: Identifier
    prior_version: str = Field(min_length=1, max_length=128)
    new_version: str = Field(min_length=1, max_length=128)
    affected_object_types: tuple[Identifier, ...] = Field(min_length=1)
    impact: CompatibilityImpact
    migration_ref: ObjectRef | None
    evidence_refs: tuple[ObjectRef, ...] = ()
    approved_review_record_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    declaration_sha256: Sha256

    @model_validator(mode="after")
    def validate_migration(self) -> CompatibilityDeclaration:
        if self.impact is CompatibilityImpact.NO_EFFECT and not self.evidence_refs:
            raise ValueError("NO_EFFECT requires compatibility evidence")
        if self.impact is not CompatibilityImpact.NO_EFFECT and self.migration_ref is None:
            raise ValueError("REVALIDATE or INVALIDATE requires migration/invalidation procedure")
        return self
