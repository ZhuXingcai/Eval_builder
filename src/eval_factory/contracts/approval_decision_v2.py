from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.approval import (
    ApprovalCheckpoint,
    ExampleKind,
    InvalidationScope,
    TypedAdjustment,
    UserDecision,
    UserDecisionRecord,
)
from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
    VersionBinding,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2

USER_DECISION_POLICY_VERSION: Literal["user-decision/r7-05-v1"] = "user-decision/r7-05-v1"


class UserApprovalRequestRevisionStateV2(StrEnum):
    MORE_EXAMPLES_REQUESTED = "MORE_EXAMPLES_REQUESTED"
    MATERIALIZED = "MATERIALIZED"


class UserDecisionCommitOutcomeV2(StrEnum):
    ACCEPTED = "ACCEPTED"
    ADJUSTMENT_ACCEPTED = "ADJUSTMENT_ACCEPTED"
    REJECTED = "REJECTED"
    MORE_EXAMPLES_REQUESTED = "MORE_EXAMPLES_REQUESTED"
    DEFERRED = "DEFERRED"


class UserDecisionHandlingPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-decision-handling-policy/v2"] = (
        "eval-factory/user-decision-handling-policy/v2"
    )
    handling_policy_id: Identifier
    max_adjustments_per_decision: int = Field(ge=1, le=10_000)
    max_environment_decisions_per_decision: int = Field(ge=1, le=100_000)
    max_effect_result_refs: int = Field(ge=1, le=100_000)
    max_request_revision_depth: int = Field(ge=1, le=10_000)
    max_reason_characters: int = Field(ge=1, le=4000)
    policy_version: Literal["user-decision/r7-05-v1"] = USER_DECISION_POLICY_VERSION
    policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _validate_audit(self.audit, (), "user decision handling policy")
        _validate_identity(
            object_id=self.handling_policy_id,
            object_sha256=self.policy_sha256,
            expected_prefix="user-decision-handling-policy",
            observed=user_decision_handling_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        max_adjustments_per_decision: int,
        max_environment_decisions_per_decision: int,
        max_effect_result_refs: int,
        max_request_revision_depth: int,
        max_reason_characters: int,
        audit: ContractAudit,
    ) -> UserDecisionHandlingPolicyV2:
        value = cls(
            handling_policy_id="user-decision-handling-policy://pending",
            max_adjustments_per_decision=max_adjustments_per_decision,
            max_environment_decisions_per_decision=(max_environment_decisions_per_decision),
            max_effect_result_refs=max_effect_result_refs,
            max_request_revision_depth=max_request_revision_depth,
            max_reason_characters=max_reason_characters,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, ()),
        )
        return _finalize(
            value,
            id_field="handling_policy_id",
            hash_field="policy_sha256",
            prefix="user-decision-handling-policy",
            digest=user_decision_handling_policy_v2_carried_sha256(value),
        )


_EFFECT_REF_TYPES = {
    ApprovalCheckpoint.LABEL_PLAN: (
        ("label-plan-adjustment-result", "v2"),
        ("label-plan", "v2"),
    ),
    ApprovalCheckpoint.TASK_REWRITE_PLAN: (
        (
            "task-rewrite-adjustment-result",
            "task-rewrite/r4-09-v1",
        ),
        ("task-rewrite-plan-version", "v2"),
    ),
    ApprovalCheckpoint.ENVIRONMENT_STRATEGY: (
        ("environment-strategy-adjustment-result", "v2"),
        ("environment-strategy", "v2"),
    ),
    ApprovalCheckpoint.FINAL_DATASET_REVIEW: (
        ("final-dataset-adjustment-result", "v2"),
        ("final-dataset-review-preview", "v2"),
    ),
}


class UserDecisionAdjustmentEffectV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-decision-adjustment-effect/v2"] = (
        "eval-factory/user-decision-adjustment-effect/v2"
    )
    effect_id: Identifier
    source_request_ref: ObjectRef
    checkpoint: ApprovalCheckpoint
    source_subject_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    source_plan_ref: ObjectRef | None
    adjustments: tuple[TypedAdjustment, ...] = Field(min_length=1)
    producer_result_ref: ObjectRef
    resulting_object_ref: ObjectRef
    related_result_refs: tuple[ObjectRef, ...] = ()
    invalidation_scope: InvalidationScope
    hard_gate_revalidation_required: Literal[True] = True
    policy_version: Literal["user-decision/r7-05-v1"] = USER_DECISION_POLICY_VERSION
    effect_sha256: Sha256
    audit: ContractAudit

    @field_validator("checkpoint", mode="before")
    @classmethod
    def parse_checkpoint(cls, value: object) -> ApprovalCheckpoint:
        return _parse_enum(value, ApprovalCheckpoint, "checkpoint")

    @model_validator(mode="after")
    def validate_effect(self) -> Self:
        _require_ref(
            self.source_request_ref,
            "user-approval-request",
            "v2",
            "source_request_ref",
        )
        _validate_checkpoint_authority(
            checkpoint=self.checkpoint,
            subject_refs=self.source_subject_refs,
            plan_ref=self.source_plan_ref,
        )
        adjustment_paths = tuple(value.target_path for value in self.adjustments)
        if adjustment_paths != tuple(sorted(adjustment_paths)) or len(adjustment_paths) != len(
            set(adjustment_paths)
        ):
            raise ValueError("effect adjustments must be sorted and unique by target path")
        producer_type, result_type = _EFFECT_REF_TYPES[self.checkpoint]
        _require_ref(
            self.producer_result_ref,
            producer_type[0],
            producer_type[1],
            "producer_result_ref",
        )
        _require_ref(
            self.resulting_object_ref,
            result_type[0],
            result_type[1],
            "resulting_object_ref",
        )
        _require_sorted_unique_refs(
            "effect related result refs",
            self.related_result_refs,
        )
        _require_safe_refs(self.related_result_refs, "related_result_refs")
        _require_sorted_unique_refs(
            "effect invalidation object refs",
            self.invalidation_scope.object_refs,
        )
        _require_safe_refs(
            self.invalidation_scope.object_refs,
            "invalidation_scope.object_refs",
        )
        if self.invalidation_scope.stages != tuple(sorted(self.invalidation_scope.stages)) or len(
            self.invalidation_scope.stages
        ) != len(set(self.invalidation_scope.stages)):
            raise ValueError("effect invalidation stages must be sorted and unique")
        _validate_audit(
            self.audit,
            _adjustment_effect_refs(self),
            "user decision adjustment effect",
        )
        _validate_identity(
            object_id=self.effect_id,
            object_sha256=self.effect_sha256,
            expected_prefix="user-decision-adjustment-effect",
            observed=user_decision_adjustment_effect_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        source_request_ref: ObjectRef,
        checkpoint: ApprovalCheckpoint,
        source_subject_refs: tuple[ObjectRef, ...],
        source_plan_ref: ObjectRef | None,
        adjustments: tuple[TypedAdjustment, ...],
        producer_result_ref: ObjectRef,
        resulting_object_ref: ObjectRef,
        related_result_refs: tuple[ObjectRef, ...],
        invalidation_scope: InvalidationScope,
        audit: ContractAudit,
    ) -> UserDecisionAdjustmentEffectV2:
        subjects = _sorted_refs(source_subject_refs)
        related = _sorted_refs(related_result_refs)
        ordered_adjustments = tuple(sorted(adjustments, key=lambda value: value.target_path))
        normalized_invalidation = InvalidationScope(
            object_refs=_sorted_refs(invalidation_scope.object_refs),
            stages=tuple(sorted(set(invalidation_scope.stages))),
        )
        refs = (
            source_request_ref,
            *subjects,
            *((source_plan_ref,) if source_plan_ref is not None else ()),
            producer_result_ref,
            resulting_object_ref,
            *related,
            *normalized_invalidation.object_refs,
        )
        value = cls(
            effect_id="user-decision-adjustment-effect://pending",
            source_request_ref=source_request_ref,
            checkpoint=checkpoint,
            source_subject_refs=subjects,
            source_plan_ref=source_plan_ref,
            adjustments=ordered_adjustments,
            producer_result_ref=producer_result_ref,
            resulting_object_ref=resulting_object_ref,
            related_result_refs=related,
            invalidation_scope=normalized_invalidation,
            effect_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            id_field="effect_id",
            hash_field="effect_sha256",
            prefix="user-decision-adjustment-effect",
            digest=user_decision_adjustment_effect_v2_carried_sha256(value),
        )


_REQUEST_EXAMPLE_KINDS = {
    ApprovalCheckpoint.LABEL_PLAN: (
        ExampleKind.POSITIVE,
        ExampleKind.NEGATIVE,
        ExampleKind.AMBIGUOUS,
        ExampleKind.ABSTAIN,
    ),
    ApprovalCheckpoint.TASK_REWRITE_PLAN: (ExampleKind.REWRITE,),
}


class UserApprovalRequestRevisionV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-approval-request-revision/v2"] = (
        "eval-factory/user-approval-request-revision/v2"
    )
    revision_id: Identifier
    job_id: Identifier
    source_request_ref: ObjectRef
    approval_policy_ref: ObjectRef
    predecessor_revision_ref: ObjectRef | None = None
    revision_number: int = Field(ge=1)
    state: UserApprovalRequestRevisionStateV2
    checkpoint: ApprovalCheckpoint
    requested_by: Identifier
    subject_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    plan_ref: ObjectRef | None
    projection_ref: ObjectRef
    required_example_kinds: tuple[ExampleKind, ...] = Field(min_length=1)
    successor_request_ref: ObjectRef | None = None
    policy_version: Literal["user-decision/r7-05-v1"] = USER_DECISION_POLICY_VERSION
    revision_sha256: Sha256
    audit: ContractAudit

    @field_validator("state", mode="before")
    @classmethod
    def parse_state(
        cls,
        value: object,
    ) -> UserApprovalRequestRevisionStateV2:
        return _parse_enum(
            value,
            UserApprovalRequestRevisionStateV2,
            "state",
        )

    @field_validator("checkpoint", mode="before")
    @classmethod
    def parse_checkpoint(cls, value: object) -> ApprovalCheckpoint:
        return _parse_enum(value, ApprovalCheckpoint, "checkpoint")

    @field_validator("required_example_kinds", mode="before")
    @classmethod
    def parse_example_kinds(
        cls,
        value: object,
    ) -> tuple[ExampleKind, ...]:
        if not isinstance(value, (tuple, list)):
            raise TypeError("required_example_kinds must be a collection")
        return tuple(item if isinstance(item, ExampleKind) else ExampleKind(item) for item in value)

    @model_validator(mode="after")
    def validate_revision(self) -> Self:
        _require_ref(
            self.source_request_ref,
            "user-approval-request",
            "v2",
            "source_request_ref",
        )
        _require_ref(
            self.approval_policy_ref,
            "user-approval-policy",
            "v2",
            "approval_policy_ref",
        )
        _validate_checkpoint_authority(
            checkpoint=self.checkpoint,
            subject_refs=self.subject_refs,
            plan_ref=self.plan_ref,
        )
        _require_ref(
            self.projection_ref,
            "user-approval-projection",
            "v2",
            "projection_ref",
        )
        expected_kinds = _REQUEST_EXAMPLE_KINDS.get(self.checkpoint)
        if expected_kinds is None or self.required_example_kinds != expected_kinds:
            raise ValueError("request revision example kinds do not match the checkpoint")
        if self.state is UserApprovalRequestRevisionStateV2.MORE_EXAMPLES_REQUESTED:
            if (
                self.revision_number != 1
                or self.predecessor_revision_ref is not None
                or self.successor_request_ref is not None
            ):
                raise ValueError("requested revision must be revision 1 without predecessor or successor")
        else:
            if (
                self.revision_number < 2
                or self.predecessor_revision_ref is None
                or self.successor_request_ref is None
            ):
                raise ValueError("materialized revision requires predecessor and successor")
            _require_ref(
                self.predecessor_revision_ref,
                "user-approval-request-revision",
                "v2",
                "predecessor_revision_ref",
            )
            _require_ref(
                self.successor_request_ref,
                "user-approval-request",
                "v2",
                "successor_request_ref",
            )
            if self.successor_request_ref == self.source_request_ref:
                raise ValueError("materialized revision requires a new successor request")
        _validate_audit(
            self.audit,
            _request_revision_refs(self),
            "user approval request revision",
        )
        _validate_identity(
            object_id=self.revision_id,
            object_sha256=self.revision_sha256,
            expected_prefix="user-approval-request-revision",
            observed=user_approval_request_revision_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create_requested(
        cls,
        *,
        job_id: str,
        source_request_ref: ObjectRef,
        approval_policy_ref: ObjectRef,
        checkpoint: ApprovalCheckpoint,
        requested_by: str,
        subject_refs: tuple[ObjectRef, ...],
        plan_ref: ObjectRef | None,
        projection_ref: ObjectRef,
        audit: ContractAudit,
    ) -> UserApprovalRequestRevisionV2:
        subjects = _sorted_refs(subject_refs)
        refs = (
            source_request_ref,
            approval_policy_ref,
            *subjects,
            *((plan_ref,) if plan_ref is not None else ()),
            projection_ref,
        )
        value = cls(
            revision_id="user-approval-request-revision://pending",
            job_id=job_id,
            source_request_ref=source_request_ref,
            approval_policy_ref=approval_policy_ref,
            revision_number=1,
            state=(UserApprovalRequestRevisionStateV2.MORE_EXAMPLES_REQUESTED),
            checkpoint=checkpoint,
            requested_by=requested_by,
            subject_refs=subjects,
            plan_ref=plan_ref,
            projection_ref=projection_ref,
            required_example_kinds=_REQUEST_EXAMPLE_KINDS.get(checkpoint, ()),
            revision_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            id_field="revision_id",
            hash_field="revision_sha256",
            prefix="user-approval-request-revision",
            digest=user_approval_request_revision_v2_carried_sha256(value),
        )

    @classmethod
    def create_materialized(
        cls,
        *,
        pending_revision: UserApprovalRequestRevisionV2,
        successor_request_ref: ObjectRef,
        audit: ContractAudit,
    ) -> UserApprovalRequestRevisionV2:
        validate_user_approval_request_revision_v2_identity(pending_revision)
        if pending_revision.state is not UserApprovalRequestRevisionStateV2.MORE_EXAMPLES_REQUESTED:
            raise ValueError("only a requested revision can be materialized")
        predecessor_ref = user_approval_request_revision_v2_ref(pending_revision)
        refs = (
            pending_revision.source_request_ref,
            pending_revision.approval_policy_ref,
            predecessor_ref,
            *pending_revision.subject_refs,
            *((pending_revision.plan_ref,) if pending_revision.plan_ref is not None else ()),
            pending_revision.projection_ref,
            successor_request_ref,
        )
        value = cls(
            revision_id="user-approval-request-revision://pending",
            job_id=pending_revision.job_id,
            source_request_ref=pending_revision.source_request_ref,
            approval_policy_ref=pending_revision.approval_policy_ref,
            predecessor_revision_ref=predecessor_ref,
            revision_number=pending_revision.revision_number + 1,
            state=UserApprovalRequestRevisionStateV2.MATERIALIZED,
            checkpoint=pending_revision.checkpoint,
            requested_by=pending_revision.requested_by,
            subject_refs=pending_revision.subject_refs,
            plan_ref=pending_revision.plan_ref,
            projection_ref=pending_revision.projection_ref,
            required_example_kinds=pending_revision.required_example_kinds,
            successor_request_ref=successor_request_ref,
            revision_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            id_field="revision_id",
            hash_field="revision_sha256",
            prefix="user-approval-request-revision",
            digest=user_approval_request_revision_v2_carried_sha256(value),
        )


_OUTCOME_DECISIONS = {
    UserDecisionCommitOutcomeV2.ACCEPTED: UserDecision.ACCEPT,
    UserDecisionCommitOutcomeV2.ADJUSTMENT_ACCEPTED: UserDecision.ADJUST,
    UserDecisionCommitOutcomeV2.REJECTED: UserDecision.REJECT,
    UserDecisionCommitOutcomeV2.MORE_EXAMPLES_REQUESTED: (UserDecision.REQUEST_MORE_EXAMPLES),
    UserDecisionCommitOutcomeV2.DEFERRED: UserDecision.DEFER,
}


class UserDecisionCommitResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-decision-commit-result/v2"] = (
        "eval-factory/user-decision-commit-result/v2"
    )
    commit_id: Identifier
    job_id: Identifier
    dataset_job_spec_ref: ObjectRef
    request_compilation_result_ref: ObjectRef
    request_ref: ObjectRef
    decision_policy_ref: ObjectRef
    authentication_context_ref: ObjectRef
    decision_record: UserDecisionRecord
    decision_record_ref: ObjectRef
    adjustment_effect: UserDecisionAdjustmentEffectV2 | None = None
    request_revision: UserApprovalRequestRevisionV2 | None = None
    outcome: UserDecisionCommitOutcomeV2
    policy_version: Literal["user-decision/r7-05-v1"] = USER_DECISION_POLICY_VERSION
    commit_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> UserDecisionCommitOutcomeV2:
        return _parse_enum(value, UserDecisionCommitOutcomeV2, "outcome")

    @model_validator(mode="after")
    def validate_commit(self) -> Self:
        _require_ref(
            self.dataset_job_spec_ref,
            "dataset-job-spec",
            "v2",
            "dataset_job_spec_ref",
        )
        _require_ref(
            self.request_compilation_result_ref,
            "user-approval-request-compilation-result",
            "v2",
            "request_compilation_result_ref",
        )
        _require_ref(
            self.request_ref,
            "user-approval-request",
            "v2",
            "request_ref",
        )
        _require_ref(
            self.decision_policy_ref,
            "user-decision-handling-policy",
            "v2",
            "decision_policy_ref",
        )
        _require_ref(
            self.authentication_context_ref,
            "authenticated-user-context",
            "v2",
            "authentication_context_ref",
        )
        expected_decision_ref = user_decision_record_ref(self.decision_record)
        if self.decision_record_ref != expected_decision_ref:
            raise ValueError("decision_record_ref does not match the nested record")
        if self.decision_record.request_ref != self.request_ref:
            raise ValueError("decision record request does not match commit request")
        if self.decision_record.decision is not _OUTCOME_DECISIONS[self.outcome]:
            raise ValueError("decision record does not match commit outcome")
        environment_ids = tuple(value.requirement_id for value in self.decision_record.environment_decisions)
        if environment_ids != tuple(sorted(environment_ids)) or len(environment_ids) != len(
            set(environment_ids)
        ):
            raise ValueError("environment decision requirement IDs must be sorted and unique")
        if self.outcome not in {
            UserDecisionCommitOutcomeV2.ACCEPTED,
            UserDecisionCommitOutcomeV2.ADJUSTMENT_ACCEPTED,
        } and (
            self.decision_record.environment_decisions or self.decision_record.query_packaging is not None
        ):
            raise ValueError("commit outcome cannot carry environment choices")
        if self.outcome is UserDecisionCommitOutcomeV2.ADJUSTMENT_ACCEPTED:
            if self.adjustment_effect is None or self.request_revision is not None:
                raise ValueError("adjustment commit requires exactly one effect")
            effect = self.adjustment_effect
            validate_user_decision_adjustment_effect_v2_identity(effect)
            if (
                effect.source_request_ref != self.request_ref
                or effect.checkpoint is not self.decision_record.checkpoint
                or effect.source_subject_refs != self.decision_record.subject_refs
                or effect.source_plan_ref != self.decision_record.plan_ref
                or effect.adjustments != self.decision_record.adjustments
                or effect.resulting_object_ref != self.decision_record.resulting_object_ref
                or effect.invalidation_scope != self.decision_record.invalidation_scope
            ):
                raise ValueError("adjustment effect does not match the decision record")
        elif self.outcome is UserDecisionCommitOutcomeV2.MORE_EXAMPLES_REQUESTED:
            if self.request_revision is None or self.adjustment_effect is not None:
                raise ValueError("more-examples commit requires exactly one request revision")
            revision = self.request_revision
            validate_user_approval_request_revision_v2_identity(revision)
            if (
                revision.source_request_ref != self.request_ref
                or revision.job_id != self.job_id
                or revision.approval_policy_ref != self.decision_record.approval_policy_ref
                or revision.checkpoint is not self.decision_record.checkpoint
                or revision.requested_by != self.decision_record.authenticated_user
                or revision.subject_refs != self.decision_record.subject_refs
                or revision.plan_ref != self.decision_record.plan_ref
                or revision.projection_ref != self.decision_record.projection_ref
                or user_approval_request_revision_v2_ref(revision)
                != self.decision_record.resulting_object_ref
            ):
                raise ValueError("request revision does not match the decision record")
        elif self.adjustment_effect is not None or self.request_revision is not None:
            raise ValueError("commit outcome cannot carry effect or request revision")
        elif self.decision_record.resulting_object_ref is not None:
            raise ValueError("commit outcome cannot carry a resulting object")
        _validate_audit(
            self.audit,
            _decision_commit_refs(self),
            "user decision commit result",
        )
        _validate_identity(
            object_id=self.commit_id,
            object_sha256=self.commit_sha256,
            expected_prefix="user-decision-commit",
            observed=user_decision_commit_result_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        dataset_job_spec_ref: ObjectRef,
        request_compilation_result_ref: ObjectRef,
        request_ref: ObjectRef,
        decision_policy_ref: ObjectRef,
        authentication_context_ref: ObjectRef,
        decision_record: UserDecisionRecord,
        adjustment_effect: UserDecisionAdjustmentEffectV2 | None,
        request_revision: UserApprovalRequestRevisionV2 | None,
        outcome: UserDecisionCommitOutcomeV2,
        audit: ContractAudit,
    ) -> UserDecisionCommitResultV2:
        decision_ref = user_decision_record_ref(decision_record)
        refs = (
            dataset_job_spec_ref,
            request_compilation_result_ref,
            request_ref,
            decision_policy_ref,
            authentication_context_ref,
            decision_ref,
            *(
                (user_decision_adjustment_effect_v2_ref(adjustment_effect),)
                if adjustment_effect is not None
                else ()
            ),
            *(
                (user_approval_request_revision_v2_ref(request_revision),)
                if request_revision is not None
                else ()
            ),
        )
        value = cls(
            commit_id="user-decision-commit://pending",
            job_id=job_id,
            dataset_job_spec_ref=dataset_job_spec_ref,
            request_compilation_result_ref=request_compilation_result_ref,
            request_ref=request_ref,
            decision_policy_ref=decision_policy_ref,
            authentication_context_ref=authentication_context_ref,
            decision_record=decision_record,
            decision_record_ref=decision_ref,
            adjustment_effect=adjustment_effect,
            request_revision=request_revision,
            outcome=outcome,
            commit_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            id_field="commit_id",
            hash_field="commit_sha256",
            prefix="user-decision-commit",
            digest=user_decision_commit_result_v2_carried_sha256(value),
        )


def user_decision_handling_policy_v2_carried_sha256(
    value: UserDecisionHandlingPolicyV2,
) -> str:
    return _carried_sha256(
        value,
        exclude={"handling_policy_id", "policy_sha256", "audit"},
    )


def user_decision_handling_policy_v2_ref(
    value: UserDecisionHandlingPolicyV2,
) -> ObjectRef:
    validate_user_decision_handling_policy_v2_identity(value)
    return ObjectRef(
        object_type="user-decision-handling-policy",
        object_id=value.handling_policy_id,
        object_version="v2",
        object_sha256=value.policy_sha256,
    )


def user_decision_adjustment_effect_v2_carried_sha256(
    value: UserDecisionAdjustmentEffectV2,
) -> str:
    return _carried_sha256(
        value,
        exclude={"effect_id", "effect_sha256", "audit"},
    )


def user_decision_adjustment_effect_v2_ref(
    value: UserDecisionAdjustmentEffectV2,
) -> ObjectRef:
    validate_user_decision_adjustment_effect_v2_identity(value)
    return ObjectRef(
        object_type="user-decision-adjustment-effect",
        object_id=value.effect_id,
        object_version="v2",
        object_sha256=value.effect_sha256,
    )


def user_approval_request_revision_v2_carried_sha256(
    value: UserApprovalRequestRevisionV2,
) -> str:
    return _carried_sha256(
        value,
        exclude={"revision_id", "revision_sha256", "audit"},
    )


def user_approval_request_revision_v2_ref(
    value: UserApprovalRequestRevisionV2,
) -> ObjectRef:
    validate_user_approval_request_revision_v2_identity(value)
    return ObjectRef(
        object_type="user-approval-request-revision",
        object_id=value.revision_id,
        object_version="v2",
        object_sha256=value.revision_sha256,
    )


def user_decision_record_carried_sha256(
    value: UserDecisionRecord,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={"decision_record_id", "record_sha256"},
            exclude_none=False,
        )
    )


def user_decision_record_ref(value: UserDecisionRecord) -> ObjectRef:
    validate_user_decision_record_identity(value)
    return ObjectRef(
        object_type="user-decision-record",
        object_id=value.decision_record_id,
        object_version="v2",
        object_sha256=value.record_sha256,
    )


def user_decision_commit_result_v2_carried_sha256(
    value: UserDecisionCommitResultV2,
) -> str:
    return _payload_sha256(
        {
            "job_id": value.job_id,
            "dataset_job_spec_ref": _ref_payload(value.dataset_job_spec_ref),
            "request_compilation_result_ref": _ref_payload(value.request_compilation_result_ref),
            "request_ref": _ref_payload(value.request_ref),
            "decision_policy_ref": _ref_payload(value.decision_policy_ref),
            "authentication_context_ref": _ref_payload(value.authentication_context_ref),
            "decision_record_ref": _ref_payload(value.decision_record_ref),
            "adjustment_effect_ref": (
                _ref_payload(user_decision_adjustment_effect_v2_ref(value.adjustment_effect))
                if value.adjustment_effect is not None
                else None
            ),
            "request_revision_ref": (
                _ref_payload(user_approval_request_revision_v2_ref(value.request_revision))
                if value.request_revision is not None
                else None
            ),
            "outcome": value.outcome.value,
            "policy_version": value.policy_version,
        }
    )


def user_decision_commit_result_v2_ref(
    value: UserDecisionCommitResultV2,
) -> ObjectRef:
    validate_user_decision_commit_result_v2_identity(value)
    return ObjectRef(
        object_type="user-decision-commit",
        object_id=value.commit_id,
        object_version="v2",
        object_sha256=value.commit_sha256,
    )


def validate_user_decision_handling_policy_v2_identity(
    value: UserDecisionHandlingPolicyV2,
) -> None:
    _validate_audit(value.audit, (), "user decision handling policy")
    _require_current_identity(
        object_id=value.handling_policy_id,
        object_sha256=value.policy_sha256,
        expected_prefix="user-decision-handling-policy",
        observed=user_decision_handling_policy_v2_carried_sha256(value),
    )


def validate_user_decision_adjustment_effect_v2_identity(
    value: UserDecisionAdjustmentEffectV2,
) -> None:
    _validate_audit(
        value.audit,
        _adjustment_effect_refs(value),
        "user decision adjustment effect",
    )
    _require_current_identity(
        object_id=value.effect_id,
        object_sha256=value.effect_sha256,
        expected_prefix="user-decision-adjustment-effect",
        observed=user_decision_adjustment_effect_v2_carried_sha256(value),
    )


def validate_user_approval_request_revision_v2_identity(
    value: UserApprovalRequestRevisionV2,
) -> None:
    _validate_audit(
        value.audit,
        _request_revision_refs(value),
        "user approval request revision",
    )
    _require_current_identity(
        object_id=value.revision_id,
        object_sha256=value.revision_sha256,
        expected_prefix="user-approval-request-revision",
        observed=user_approval_request_revision_v2_carried_sha256(value),
    )


def validate_user_decision_record_identity(
    value: UserDecisionRecord,
) -> None:
    observed = user_decision_record_carried_sha256(value)
    _require_current_identity(
        object_id=value.decision_record_id,
        object_sha256=value.record_sha256,
        expected_prefix="user-decision-record",
        observed=observed,
    )


def validate_user_decision_commit_result_v2_identity(
    value: UserDecisionCommitResultV2,
) -> None:
    _validate_audit(
        value.audit,
        _decision_commit_refs(value),
        "user decision commit result",
    )
    _require_current_identity(
        object_id=value.commit_id,
        object_sha256=value.commit_sha256,
        expected_prefix="user-decision-commit",
        observed=user_decision_commit_result_v2_carried_sha256(value),
    )


def _adjustment_effect_refs(
    value: UserDecisionAdjustmentEffectV2,
) -> tuple[ObjectRef, ...]:
    return (
        value.source_request_ref,
        *value.source_subject_refs,
        *((value.source_plan_ref,) if value.source_plan_ref is not None else ()),
        value.producer_result_ref,
        value.resulting_object_ref,
        *value.related_result_refs,
        *value.invalidation_scope.object_refs,
    )


def _request_revision_refs(
    value: UserApprovalRequestRevisionV2,
) -> tuple[ObjectRef, ...]:
    return (
        value.source_request_ref,
        value.approval_policy_ref,
        *((value.predecessor_revision_ref,) if value.predecessor_revision_ref is not None else ()),
        *value.subject_refs,
        *((value.plan_ref,) if value.plan_ref is not None else ()),
        value.projection_ref,
        *((value.successor_request_ref,) if value.successor_request_ref is not None else ()),
    )


def _decision_commit_refs(
    value: UserDecisionCommitResultV2,
) -> tuple[ObjectRef, ...]:
    return (
        value.dataset_job_spec_ref,
        value.request_compilation_result_ref,
        value.request_ref,
        value.decision_policy_ref,
        value.authentication_context_ref,
        value.decision_record_ref,
        *(
            (user_decision_adjustment_effect_v2_ref(value.adjustment_effect),)
            if value.adjustment_effect is not None
            else ()
        ),
        *(
            (user_approval_request_revision_v2_ref(value.request_revision),)
            if value.request_revision is not None
            else ()
        ),
    )


def _validate_checkpoint_authority(
    *,
    checkpoint: ApprovalCheckpoint,
    subject_refs: tuple[ObjectRef, ...],
    plan_ref: ObjectRef | None,
) -> None:
    _require_sorted_unique_refs("checkpoint subject refs", subject_refs)
    _require_safe_refs(subject_refs, "subject_refs")
    if checkpoint is ApprovalCheckpoint.LABEL_PLAN:
        if len(subject_refs) != 1:
            raise ValueError("LABEL_PLAN requires one label-spec subject")
        _require_ref(subject_refs[0], "label-spec", "v2", "subject_refs")
        if plan_ref is None:
            raise ValueError("LABEL_PLAN requires a label-plan")
        _require_ref(plan_ref, "label-plan", "v2", "plan_ref")
        return
    if checkpoint is ApprovalCheckpoint.TASK_REWRITE_PLAN:
        if len(subject_refs) != 1:
            raise ValueError("TASK_REWRITE_PLAN requires one selection-context subject")
        _require_ref(
            subject_refs[0],
            "selection-context",
            "v2",
            "subject_refs",
        )
        if plan_ref is None:
            raise ValueError("TASK_REWRITE_PLAN requires a plan version")
        _require_ref(
            plan_ref,
            "task-rewrite-plan-version",
            "v2",
            "plan_ref",
        )
        return
    if checkpoint is ApprovalCheckpoint.ENVIRONMENT_STRATEGY:
        for ref in subject_refs:
            _require_ref(ref, "task-draft", "v2", "subject_refs")
        if plan_ref is None:
            raise ValueError("ENVIRONMENT_STRATEGY requires a strategy")
        _require_ref(plan_ref, "environment-strategy", "v2", "plan_ref")
        return
    for ref in subject_refs:
        _require_ref(ref, "evaluation-item", "v2", "subject_refs")
    if plan_ref is not None:
        raise ValueError("FINAL_DATASET_REVIEW cannot bind a plan")


def _carried_sha256(
    value: ContractModelV2,
    *,
    exclude: set[str],
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude=exclude,
            exclude_none=False,
        )
    )


def _payload_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            canonical_value_v2(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _ref_payload(value: ObjectRef) -> dict[str, object]:
    return value.model_dump(mode="json", exclude_none=False)


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    unique = {_ref_key(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


def _require_sorted_unique_refs(
    label: str,
    values: tuple[ObjectRef, ...],
) -> None:
    keys = tuple(_ref_key(value) for value in values)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"{label} must be sorted and unique")


def _require_safe_refs(
    values: tuple[ObjectRef, ...],
    label: str,
) -> None:
    denied = {
        "credential",
        "final-answer",
        "final-output",
        "grader-rule",
        "hidden-pass-condition",
        "hidden-selection-signal",
        "private-reference",
        "quarantine",
        "raw-trace",
        "raw-traj",
        "restricted-trace-span",
        "secret",
    }
    if any(ref.object_type in denied for ref in values):
        raise ValueError(f"{label} contain a restricted reference")


def _require_ref(
    ref: ObjectRef,
    expected_type: str,
    expected_version: str,
    field_name: str,
) -> None:
    if ref.object_type != expected_type or ref.object_version != expected_version:
        raise ValueError(f"{field_name} must reference {expected_type} {expected_version}")


def _validate_audit(
    audit: ContractAudit,
    expected_refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != _sorted_refs(expected_refs):
        raise ValueError(f"{label} audit input refs are not exact")
    bindings = tuple(value for value in audit.governing_versions if value.component == "user-decision")
    if len(bindings) != 1 or bindings[0].version != USER_DECISION_POLICY_VERSION:
        raise ValueError(f"{label} audit is missing the current decision policy")


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    governing = (
        *(value for value in audit.governing_versions if value.component != "user-decision"),
        VersionBinding(
            component="user-decision",
            version=USER_DECISION_POLICY_VERSION,
        ),
    )
    return audit.model_copy(
        update={
            "governing_versions": tuple(
                sorted(
                    governing,
                    key=lambda value: (
                        value.component,
                        value.version,
                        value.sha256 or "",
                    ),
                )
            ),
            "input_refs": _sorted_refs(refs),
        }
    )


def _validate_identity(
    *,
    object_id: str,
    object_sha256: str,
    expected_prefix: str,
    observed: str,
) -> None:
    if object_id.endswith("://pending") or object_sha256 == "0" * 64:
        return
    if object_sha256 != observed or object_id != f"{expected_prefix}://sha256/{observed}":
        raise ValueError(f"{expected_prefix} identity is stale")


def _require_current_identity(
    *,
    object_id: str,
    object_sha256: str,
    expected_prefix: str,
    observed: str,
) -> None:
    if object_id.endswith("://pending") or object_sha256 == "0" * 64:
        raise ValueError(f"{expected_prefix} identity is pending")
    _validate_identity(
        object_id=object_id,
        object_sha256=object_sha256,
        expected_prefix=expected_prefix,
        observed=observed,
    )


def _finalize[ModelT: ContractModelV2](
    value: ModelT,
    *,
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
