from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from eval_factory.contracts.approval import ApprovalCheckpoint
from eval_factory.contracts.core import ContractAudit, Identifier, ObjectRef, Sha256
from eval_factory.contracts.core_v2 import ContractModelV2
from eval_factory.contracts.release import EvaluationItemComponents, ReleaseChannel


class ReleaseStateV2(StrEnum):
    CANDIDATE = "CANDIDATE"
    REJECTED = "REJECTED"
    APPROVED = "APPROVED"
    RELEASED = "RELEASED"
    REVOKED = "REVOKED"


class ReleaseActionV2(StrEnum):
    REQUEST_RELEASE = "REQUEST_RELEASE"
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    PUBLISH = "PUBLISH"
    REVOKE = "REVOKE"


class CheckpointDecisionBinding(ContractModelV2):
    schema_version: Literal["eval-factory/checkpoint-decision-binding/v2"] = (
        "eval-factory/checkpoint-decision-binding/v2"
    )
    checkpoint: ApprovalCheckpoint
    request_ref: ObjectRef
    decision_record_ref: ObjectRef


class ReleaseDecisionV2(ContractModelV2):
    schema_version: Literal["eval-factory/release-decision/v2"] = "eval-factory/release-decision/v2"
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
    automated_quality_passed: bool
    open_p0_count: int = Field(ge=0)
    open_p1_count: int = Field(ge=0)
    unresolved_non_waivable_count: int = Field(ge=0)
    user_approval_policy_ref: ObjectRef
    required_checkpoints: frozenset[ApprovalCheckpoint]
    checkpoint_decisions: tuple[CheckpointDecisionBinding, ...] = ()
    channel: ReleaseChannel
    registry: Identifier
    export_profile: Literal["LH", "GENERIC"]
    export_profile_version: str = Field(min_length=1, max_length=128)
    production_attestation_ref: ObjectRef | None
    action: ReleaseActionV2
    state: ReleaseStateV2
    idempotency_key: Identifier
    actor: Identifier
    decided_at: datetime
    decision_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_release(self) -> ReleaseDecisionV2:
        if self.channel is ReleaseChannel.PRODUCTION:
            if self.production_attestation_ref is None:
                raise ValueError("production decision requires readiness attestation")
        elif self.production_attestation_ref is not None:
            raise ValueError("non-production decision must not bind production attestation")

        action_states = {
            ReleaseActionV2.REQUEST_RELEASE: ReleaseStateV2.CANDIDATE,
            ReleaseActionV2.APPROVE: ReleaseStateV2.APPROVED,
            ReleaseActionV2.REJECT: ReleaseStateV2.REJECTED,
            ReleaseActionV2.PUBLISH: ReleaseStateV2.RELEASED,
            ReleaseActionV2.REVOKE: ReleaseStateV2.REVOKED,
        }
        if self.state is not action_states[self.action]:
            raise ValueError(f"{self.action} action must create {action_states[self.action]} state")

        bound_checkpoints = [binding.checkpoint for binding in self.checkpoint_decisions]
        if len(set(bound_checkpoints)) != len(bound_checkpoints):
            raise ValueError("a checkpoint can bind at most one current UserDecisionRecord")
        satisfied_checkpoints = frozenset(bound_checkpoints)
        if not self.required_checkpoints and self.checkpoint_decisions:
            raise ValueError("disabled checkpoints cannot create synthetic decisions")
        if not satisfied_checkpoints.issubset(self.required_checkpoints):
            raise ValueError("checkpoint decisions must be required by UserApprovalPolicy")

        if self.action in {ReleaseActionV2.APPROVE, ReleaseActionV2.PUBLISH}:
            blockers = self.open_p0_count + self.open_p1_count + self.unresolved_non_waivable_count
            if not self.automated_quality_passed or blockers:
                raise ValueError("approval and publication require passing automated hard gates")
            if satisfied_checkpoints != self.required_checkpoints:
                raise ValueError("all required user checkpoints must be current")
        return self


class EvaluationItemV2(ContractModelV2):
    schema_version: Literal["eval-factory/evaluation-item/v2"] = "eval-factory/evaluation-item/v2"
    evaluation_item_id: Identifier
    item_version: str = Field(min_length=1, max_length=64)
    source_trace_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    label_decision_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    task_draft_ref: ObjectRef
    attachment_reconstruction_result_ref: ObjectRef
    components: EvaluationItemComponents
    user_approval_policy_ref: ObjectRef
    user_decision_record_refs: tuple[ObjectRef, ...] = ()
    release_decision_ref: ObjectRef
    item_sha256: Sha256
    audit: ContractAudit
