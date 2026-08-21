from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.approval import (
    ALL_CHECKPOINTS,
    ApprovalCheckpoint,
    EnvironmentScopeDecision,
    EnvironmentStrategy,
    LabelPlan,
    QueryPackagingChoice,
    TypedAdjustment,
    UserApprovalRequest,
    UserDecision,
)
from eval_factory.contracts.approval_decision_v2 import (
    UserDecisionAdjustmentEffectV2,
    UserDecisionCommitResultV2,
    user_decision_commit_result_v2_ref,
)
from eval_factory.contracts.approval_v2 import (
    FinalDatasetReviewPreviewV2,
    environment_strategy_ref,
    final_dataset_review_preview_v2_ref,
    label_plan_ref,
    user_approval_request_ref,
)
from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import (
    ContractModelV2,
    canonical_value_v2,
)
from eval_factory.contracts.orchestration import JobStatus
from eval_factory.contracts.task_v2 import (
    TaskRewritePlanPreviewV2,
    task_rewrite_plan_preview_ref,
)

CHECKPOINT_INTERACTION_POLICY_VERSION: Literal["checkpoint-interaction/r7-07-v1"] = (
    "checkpoint-interaction/r7-07-v1"
)


class UserCheckpointInteractionStateV2(StrEnum):
    PENDING_DECISION = "PENDING_DECISION"
    RESUMABLE = "RESUMABLE"
    WAITING_REVISION = "WAITING_REVISION"
    WAITING_REVALIDATION = "WAITING_REVALIDATION"
    DEFERRED = "DEFERRED"
    RESUMED = "RESUMED"
    TERMINATED = "TERMINATED"
    SUPERSEDED = "SUPERSEDED"


class UserCheckpointOpenOutcomeV2(StrEnum):
    PAUSED = "PAUSED"
    DISABLED = "DISABLED"
    NOT_REQUIRED = "NOT_REQUIRED"


class UserCheckpointResumeDispositionV2(StrEnum):
    DIRECT = "DIRECT"
    AFTER_REVALIDATION = "AFTER_REVALIDATION"
    NEXT_REQUEST = "NEXT_REQUEST"
    WAITING_REVISION = "WAITING_REVISION"
    DEFERRED = "DEFERRED"
    TERMINATED = "TERMINATED"


class UserCheckpointInteractionPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-checkpoint-interaction-policy/v2"] = (
        "eval-factory/user-checkpoint-interaction-policy/v2"
    )
    policy_id: Identifier
    max_interactions_per_job: int = Field(ge=1, le=1_000_000)
    max_requests_per_interaction: int = Field(ge=1, le=100_000)
    max_page_size: int = Field(ge=1, le=10_000)
    max_presentation_bytes: int = Field(ge=2, le=64 * 1024 * 1024)
    max_source_context_bytes: int = Field(ge=2, le=256 * 1024 * 1024)
    max_ready_work_refs: int = Field(ge=1, le=100_000)
    allowed_checkpoints: frozenset[ApprovalCheckpoint]
    policy_version: Literal["checkpoint-interaction/r7-07-v1"] = CHECKPOINT_INTERACTION_POLICY_VERSION
    policy_sha256: Sha256
    audit: ContractAudit

    @field_validator("allowed_checkpoints", mode="before")
    @classmethod
    def parse_checkpoints(
        cls,
        value: object,
    ) -> frozenset[ApprovalCheckpoint]:
        if not isinstance(value, (set, frozenset, list, tuple)):
            raise TypeError("allowed_checkpoints must be a checkpoint collection")
        return frozenset(
            item if isinstance(item, ApprovalCheckpoint) else ApprovalCheckpoint(item) for item in value
        )

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.allowed_checkpoints != ALL_CHECKPOINTS:
            raise ValueError("interaction policy must allow the exact R7 checkpoint set")
        _validate_audit(self.audit, (), "checkpoint interaction policy")
        _validate_identity(
            self.policy_id,
            self.policy_sha256,
            "user-checkpoint-interaction-policy",
            user_checkpoint_interaction_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        max_interactions_per_job: int,
        max_requests_per_interaction: int,
        max_page_size: int,
        max_presentation_bytes: int,
        max_source_context_bytes: int,
        max_ready_work_refs: int,
        audit: ContractAudit,
    ) -> UserCheckpointInteractionPolicyV2:
        value = cls(
            policy_id="user-checkpoint-interaction-policy://pending",
            max_interactions_per_job=max_interactions_per_job,
            max_requests_per_interaction=max_requests_per_interaction,
            max_page_size=max_page_size,
            max_presentation_bytes=max_presentation_bytes,
            max_source_context_bytes=max_source_context_bytes,
            max_ready_work_refs=max_ready_work_refs,
            allowed_checkpoints=ALL_CHECKPOINTS,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, ()),
        )
        return _round_trip(
            _finalize(
                value,
                "policy_id",
                "policy_sha256",
                "user-checkpoint-interaction-policy",
                user_checkpoint_interaction_policy_v2_carried_sha256(value),
            )
        )


class UserCheckpointPresentationV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-checkpoint-presentation/v2"] = (
        "eval-factory/user-checkpoint-presentation/v2"
    )
    presentation_id: Identifier
    job_id: Identifier
    request_ref: ObjectRef
    checkpoint: ApprovalCheckpoint
    label_plan: LabelPlan | None = None
    task_rewrite_preview: TaskRewritePlanPreviewV2 | None = None
    environment_strategy: EnvironmentStrategy | None = None
    final_dataset_review_preview: FinalDatasetReviewPreviewV2 | None = None
    presentation_sha256: Sha256
    audit: ContractAudit

    @field_validator("checkpoint", mode="before")
    @classmethod
    def parse_checkpoint(cls, value: object) -> ApprovalCheckpoint:
        if isinstance(value, ApprovalCheckpoint):
            return value
        if isinstance(value, str):
            return ApprovalCheckpoint(value)
        raise TypeError("checkpoint must be an ApprovalCheckpoint")

    @model_validator(mode="after")
    def validate_presentation(self) -> Self:
        _require_ref(
            self.request_ref,
            "user-approval-request",
            "v2",
            "request_ref",
        )
        bodies = (
            self.label_plan,
            self.task_rewrite_preview,
            self.environment_strategy,
            self.final_dataset_review_preview,
        )
        if sum(value is not None for value in bodies) != 1:
            raise ValueError("checkpoint presentation requires exactly one body")
        expected_ref = {
            ApprovalCheckpoint.LABEL_PLAN: (
                label_plan_ref(self.label_plan) if self.label_plan is not None else None
            ),
            ApprovalCheckpoint.TASK_REWRITE_PLAN: (
                task_rewrite_plan_preview_ref(self.task_rewrite_preview)
                if self.task_rewrite_preview is not None
                else None
            ),
            ApprovalCheckpoint.ENVIRONMENT_STRATEGY: (
                environment_strategy_ref(self.environment_strategy)
                if self.environment_strategy is not None
                else None
            ),
            ApprovalCheckpoint.FINAL_DATASET_REVIEW: (
                final_dataset_review_preview_v2_ref(self.final_dataset_review_preview)
                if self.final_dataset_review_preview is not None
                else None
            ),
        }[self.checkpoint]
        if expected_ref is None:
            raise ValueError("presentation body does not match checkpoint")
        _validate_audit(
            self.audit,
            (self.request_ref, expected_ref),
            "checkpoint presentation",
        )
        _validate_identity(
            self.presentation_id,
            self.presentation_sha256,
            "user-checkpoint-presentation",
            user_checkpoint_presentation_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        request: UserApprovalRequest,
        label_plan: LabelPlan | None,
        task_rewrite_preview: TaskRewritePlanPreviewV2 | None,
        environment_strategy: EnvironmentStrategy | None,
        final_dataset_review_preview: FinalDatasetReviewPreviewV2 | None,
        audit: ContractAudit,
    ) -> UserCheckpointPresentationV2:
        request_ref = user_approval_request_ref(request)
        body_ref = _presentation_body_ref(
            request.checkpoint,
            label_plan=label_plan,
            task_rewrite_preview=task_rewrite_preview,
            environment_strategy=environment_strategy,
            final_dataset_review_preview=final_dataset_review_preview,
        )
        if request.preview_refs != (body_ref,):
            raise ValueError("presentation body differs from request preview authority")
        value = cls(
            presentation_id="user-checkpoint-presentation://pending",
            job_id=job_id,
            request_ref=request_ref,
            checkpoint=request.checkpoint,
            label_plan=label_plan,
            task_rewrite_preview=task_rewrite_preview,
            environment_strategy=environment_strategy,
            final_dataset_review_preview=final_dataset_review_preview,
            presentation_sha256="0" * 64,
            audit=_safe_audit(audit, (request_ref, body_ref)),
        )
        return _round_trip(
            _finalize(
                value,
                "presentation_id",
                "presentation_sha256",
                "user-checkpoint-presentation",
                user_checkpoint_presentation_v2_carried_sha256(value),
            )
        )


class UserCheckpointInteractionV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-checkpoint-interaction/v2"] = (
        "eval-factory/user-checkpoint-interaction/v2"
    )
    interaction_id: Identifier
    job_id: Identifier
    chain_id: Identifier
    interaction_version: int = Field(ge=1)
    predecessor_interaction_ref: ObjectRef | None = None
    checkpoint: ApprovalCheckpoint
    request_compilation_ref: ObjectRef
    request_ref: ObjectRef
    presentation_ref: ObjectRef
    source_context_ref: ObjectRef
    state: UserCheckpointInteractionStateV2
    decision_commit_ref: ObjectRef | None = None
    application_ref: ObjectRef | None = None
    job_status: JobStatus
    occurred_at: datetime
    policy_ref: ObjectRef
    interaction_sha256: Sha256
    audit: ContractAudit

    @field_validator("checkpoint", mode="before")
    @classmethod
    def parse_checkpoint(cls, value: object) -> ApprovalCheckpoint:
        if isinstance(value, ApprovalCheckpoint):
            return value
        if isinstance(value, str):
            return ApprovalCheckpoint(value)
        raise TypeError("checkpoint must be an ApprovalCheckpoint")

    @field_validator("state", mode="before")
    @classmethod
    def parse_state(
        cls,
        value: object,
    ) -> UserCheckpointInteractionStateV2:
        if isinstance(value, UserCheckpointInteractionStateV2):
            return value
        if isinstance(value, str):
            return UserCheckpointInteractionStateV2(value)
        raise TypeError("state must be a UserCheckpointInteractionStateV2")

    @model_validator(mode="after")
    def validate_interaction(self) -> Self:
        _require_ref(
            self.request_compilation_ref,
            "user-approval-request-compilation-result",
            "v2",
            "request_compilation_ref",
        )
        _require_ref(self.request_ref, "user-approval-request", "v2", "request_ref")
        _require_ref(
            self.presentation_ref,
            "user-checkpoint-presentation",
            "v2",
            "presentation_ref",
        )
        _require_ref(
            self.source_context_ref,
            "user-checkpoint-source-context",
            "private-v1",
            "source_context_ref",
        )
        _require_ref(
            self.policy_ref,
            "user-checkpoint-interaction-policy",
            "v2",
            "policy_ref",
        )
        if self.interaction_version == 1:
            if self.predecessor_interaction_ref is not None:
                raise ValueError("first interaction cannot have a predecessor")
        else:
            _require_ref(
                self.predecessor_interaction_ref,
                "user-checkpoint-interaction",
                "v2",
                "predecessor_interaction_ref",
            )
        if self.state is UserCheckpointInteractionStateV2.PENDING_DECISION:
            if self.decision_commit_ref is not None or self.application_ref is not None:
                raise ValueError("pending interaction cannot carry decision authority")
        else:
            _require_ref(
                self.decision_commit_ref,
                "user-decision-commit",
                "v2",
                "decision_commit_ref",
            )
        if self.application_ref is not None:
            _require_ref(
                self.application_ref,
                "user-plan-application",
                "v2",
                "application_ref",
            )
        expected_status = {
            UserCheckpointInteractionStateV2.PENDING_DECISION: JobStatus.BLOCKED,
            UserCheckpointInteractionStateV2.RESUMABLE: JobStatus.BLOCKED,
            UserCheckpointInteractionStateV2.WAITING_REVISION: JobStatus.BLOCKED,
            UserCheckpointInteractionStateV2.WAITING_REVALIDATION: JobStatus.BLOCKED,
            UserCheckpointInteractionStateV2.DEFERRED: JobStatus.BLOCKED,
            UserCheckpointInteractionStateV2.RESUMED: JobStatus.RUNNING,
            UserCheckpointInteractionStateV2.TERMINATED: JobStatus.FAILED,
            UserCheckpointInteractionStateV2.SUPERSEDED: JobStatus.BLOCKED,
        }[self.state]
        if self.job_status is not expected_status:
            raise ValueError("interaction state and Job status disagree")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("interaction timestamp must be timezone-aware")
        refs = _interaction_refs(self)
        _validate_audit(self.audit, refs, "checkpoint interaction")
        _validate_identity(
            self.interaction_id,
            self.interaction_sha256,
            "user-checkpoint-interaction",
            user_checkpoint_interaction_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        chain_id: str,
        interaction_version: int,
        predecessor_interaction_ref: ObjectRef | None,
        checkpoint: ApprovalCheckpoint,
        request_compilation_ref: ObjectRef,
        request_ref: ObjectRef,
        presentation_ref: ObjectRef,
        source_context_ref: ObjectRef,
        state: UserCheckpointInteractionStateV2,
        decision_commit_ref: ObjectRef | None,
        application_ref: ObjectRef | None,
        job_status: JobStatus,
        occurred_at: datetime,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> UserCheckpointInteractionV2:
        value = cls(
            interaction_id="user-checkpoint-interaction://pending",
            job_id=job_id,
            chain_id=chain_id,
            interaction_version=interaction_version,
            predecessor_interaction_ref=predecessor_interaction_ref,
            checkpoint=checkpoint,
            request_compilation_ref=request_compilation_ref,
            request_ref=request_ref,
            presentation_ref=presentation_ref,
            source_context_ref=source_context_ref,
            state=state,
            decision_commit_ref=decision_commit_ref,
            application_ref=application_ref,
            job_status=job_status,
            occurred_at=occurred_at,
            policy_ref=policy_ref,
            interaction_sha256="0" * 64,
            audit=_safe_audit(
                audit,
                tuple(
                    ref
                    for ref in (
                        predecessor_interaction_ref,
                        request_compilation_ref,
                        request_ref,
                        presentation_ref,
                        source_context_ref,
                        decision_commit_ref,
                        application_ref,
                        policy_ref,
                    )
                    if ref is not None
                ),
            ),
        )
        return _round_trip(
            _finalize(
                value,
                "interaction_id",
                "interaction_sha256",
                "user-checkpoint-interaction",
                user_checkpoint_interaction_v2_carried_sha256(value),
            )
        )


class UserCheckpointOpenResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-checkpoint-open-result/v2"] = (
        "eval-factory/user-checkpoint-open-result/v2"
    )
    result_id: Identifier
    job_id: Identifier
    outcome: UserCheckpointOpenOutcomeV2
    interaction: UserCheckpointInteractionV2 | None = None
    job_status: JobStatus
    job_version: int = Field(ge=0)
    policy_ref: ObjectRef
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> UserCheckpointOpenOutcomeV2:
        if isinstance(value, UserCheckpointOpenOutcomeV2):
            return value
        if isinstance(value, str):
            return UserCheckpointOpenOutcomeV2(value)
        raise TypeError("outcome must be a UserCheckpointOpenOutcomeV2")

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.policy_ref,
            "user-checkpoint-interaction-policy",
            "v2",
            "policy_ref",
        )
        paused = self.outcome is UserCheckpointOpenOutcomeV2.PAUSED
        if paused != (self.interaction is not None):
            raise ValueError("PAUSED outcome and interaction must be paired")
        expected_status = JobStatus.BLOCKED if paused else JobStatus.RUNNING
        if self.job_status is not expected_status:
            raise ValueError("open outcome and Job status disagree")
        if self.interaction is not None and (
            self.interaction.job_id != self.job_id
            or self.interaction.state is not UserCheckpointInteractionStateV2.PENDING_DECISION
        ):
            raise ValueError("open result interaction is invalid")
        refs = (
            *((user_checkpoint_interaction_v2_ref(self.interaction),) if self.interaction else ()),
            self.policy_ref,
        )
        _validate_audit(self.audit, refs, "checkpoint open result")
        _validate_identity(
            self.result_id,
            self.result_sha256,
            "user-checkpoint-open-result",
            user_checkpoint_open_result_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        outcome: UserCheckpointOpenOutcomeV2,
        interaction: UserCheckpointInteractionV2 | None,
        job_status: JobStatus,
        job_version: int,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> UserCheckpointOpenResultV2:
        refs = (
            *((user_checkpoint_interaction_v2_ref(interaction),) if interaction else ()),
            policy_ref,
        )
        value = cls(
            result_id="user-checkpoint-open-result://pending",
            job_id=job_id,
            outcome=outcome,
            interaction=interaction,
            job_status=job_status,
            job_version=job_version,
            policy_ref=policy_ref,
            result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _round_trip(
            _finalize(
                value,
                "result_id",
                "result_sha256",
                "user-checkpoint-open-result",
                user_checkpoint_open_result_v2_carried_sha256(value),
            )
        )


class UserCheckpointDecisionSubmissionV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-checkpoint-decision-submission/v2"] = (
        "eval-factory/user-checkpoint-decision-submission/v2"
    )
    interaction_ref: ObjectRef
    decision: UserDecision
    adjustments: tuple[TypedAdjustment, ...] = ()
    adjustment_effect: UserDecisionAdjustmentEffectV2 | None = None
    environment_decisions: tuple[EnvironmentScopeDecision, ...] = ()
    query_packaging: QueryPackagingChoice | None = None
    reason: str = Field(min_length=1, max_length=4000)
    idempotency_key: Identifier

    @field_validator("decision", mode="before")
    @classmethod
    def parse_decision(cls, value: object) -> UserDecision:
        if isinstance(value, UserDecision):
            return value
        if isinstance(value, str):
            return UserDecision(value)
        raise TypeError("decision must be a UserDecision")

    @field_validator("query_packaging", mode="before")
    @classmethod
    def parse_query_packaging(
        cls,
        value: object,
    ) -> QueryPackagingChoice | None:
        if value is None or isinstance(value, QueryPackagingChoice):
            return value
        if isinstance(value, str):
            return QueryPackagingChoice(value)
        raise TypeError("query_packaging must be a QueryPackagingChoice")

    @model_validator(mode="after")
    def validate_submission(self) -> Self:
        _require_ref(
            self.interaction_ref,
            "user-checkpoint-interaction",
            "v2",
            "interaction_ref",
        )
        paths = tuple(value.target_path for value in self.adjustments)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("adjustments must be sorted and unique")
        requirements = tuple(value.requirement_id for value in self.environment_decisions)
        if requirements != tuple(sorted(requirements)) or len(requirements) != len(set(requirements)):
            raise ValueError("environment decisions must be sorted and unique")
        if self.decision is UserDecision.ADJUST:
            if not self.adjustments or self.adjustment_effect is None:
                raise ValueError("ADJUST requires adjustments and an exact effect")
        elif self.adjustments or self.adjustment_effect is not None:
            raise ValueError("only ADJUST may carry adjustment fields")
        if self.decision not in {UserDecision.ACCEPT, UserDecision.ADJUST} and (
            self.environment_decisions or self.query_packaging is not None
        ):
            raise ValueError("only acceptance or adjustment may carry environment choices")
        return self


class UserCheckpointDecisionResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-checkpoint-decision-result/v2"] = (
        "eval-factory/user-checkpoint-decision-result/v2"
    )
    result_id: Identifier
    predecessor_interaction_ref: ObjectRef
    interaction: UserCheckpointInteractionV2
    next_interaction: UserCheckpointInteractionV2 | None = None
    decision_commit: UserDecisionCommitResultV2
    resume_disposition: UserCheckpointResumeDispositionV2
    job_status: JobStatus
    job_version: int = Field(ge=0)
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("resume_disposition", mode="before")
    @classmethod
    def parse_disposition(
        cls,
        value: object,
    ) -> UserCheckpointResumeDispositionV2:
        if isinstance(value, UserCheckpointResumeDispositionV2):
            return value
        if isinstance(value, str):
            return UserCheckpointResumeDispositionV2(value)
        raise TypeError("resume_disposition must be a UserCheckpointResumeDispositionV2")

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.predecessor_interaction_ref,
            "user-checkpoint-interaction",
            "v2",
            "predecessor_interaction_ref",
        )
        commit_ref = user_decision_commit_result_v2_ref(self.decision_commit)
        if (
            self.interaction.predecessor_interaction_ref != self.predecessor_interaction_ref
            or self.interaction.decision_commit_ref != commit_ref
            or self.interaction.job_status is not self.job_status
            or self.decision_commit.job_id != self.interaction.job_id
            or self.decision_commit.request_ref != self.interaction.request_ref
            or self.decision_commit.decision_record.checkpoint is not self.interaction.checkpoint
        ):
            raise ValueError("decision result interaction binding is stale")
        if self.next_interaction is not None and (
            self.next_interaction.predecessor_interaction_ref
            != user_checkpoint_interaction_v2_ref(self.interaction)
            or self.next_interaction.chain_id != self.interaction.chain_id
            or self.next_interaction.interaction_version != self.interaction.interaction_version + 1
            or self.next_interaction.state is not UserCheckpointInteractionStateV2.PENDING_DECISION
            or self.next_interaction.job_status is not JobStatus.BLOCKED
        ):
            raise ValueError("next checkpoint interaction is not a valid successor")
        expected = {
            UserCheckpointInteractionStateV2.RESUMABLE: (UserCheckpointResumeDispositionV2.DIRECT),
            UserCheckpointInteractionStateV2.WAITING_REVALIDATION: (
                UserCheckpointResumeDispositionV2.AFTER_REVALIDATION
            ),
            UserCheckpointInteractionStateV2.WAITING_REVISION: (
                UserCheckpointResumeDispositionV2.WAITING_REVISION
            ),
            UserCheckpointInteractionStateV2.DEFERRED: (UserCheckpointResumeDispositionV2.DEFERRED),
            UserCheckpointInteractionStateV2.TERMINATED: (UserCheckpointResumeDispositionV2.TERMINATED),
        }.get(self.interaction.state)
        if self.next_interaction is not None:
            expected = UserCheckpointResumeDispositionV2.NEXT_REQUEST
        if self.resume_disposition is not expected:
            raise ValueError("decision result disposition is not derived from state")
        refs = (
            self.predecessor_interaction_ref,
            user_checkpoint_interaction_v2_ref(self.interaction),
            *(
                (user_checkpoint_interaction_v2_ref(self.next_interaction),)
                if self.next_interaction is not None
                else ()
            ),
            commit_ref,
        )
        _validate_audit(self.audit, refs, "checkpoint decision result")
        _validate_identity(
            self.result_id,
            self.result_sha256,
            "user-checkpoint-decision-result",
            user_checkpoint_decision_result_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        predecessor_interaction_ref: ObjectRef,
        interaction: UserCheckpointInteractionV2,
        next_interaction: UserCheckpointInteractionV2 | None,
        decision_commit: UserDecisionCommitResultV2,
        resume_disposition: UserCheckpointResumeDispositionV2,
        job_status: JobStatus,
        job_version: int,
        audit: ContractAudit,
    ) -> UserCheckpointDecisionResultV2:
        refs = (
            predecessor_interaction_ref,
            user_checkpoint_interaction_v2_ref(interaction),
            *(
                (user_checkpoint_interaction_v2_ref(next_interaction),)
                if next_interaction is not None
                else ()
            ),
            user_decision_commit_result_v2_ref(decision_commit),
        )
        value = cls(
            result_id="user-checkpoint-decision-result://pending",
            predecessor_interaction_ref=predecessor_interaction_ref,
            interaction=interaction,
            next_interaction=next_interaction,
            decision_commit=decision_commit,
            resume_disposition=resume_disposition,
            job_status=job_status,
            job_version=job_version,
            result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _round_trip(
            _finalize(
                value,
                "result_id",
                "result_sha256",
                "user-checkpoint-decision-result",
                user_checkpoint_decision_result_v2_carried_sha256(value),
            )
        )


class UserCheckpointResumeResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-checkpoint-resume-result/v2"] = (
        "eval-factory/user-checkpoint-resume-result/v2"
    )
    result_id: Identifier
    predecessor_interaction_ref: ObjectRef
    interaction: UserCheckpointInteractionV2
    decision_commit_ref: ObjectRef
    application_ref: ObjectRef | None = None
    revalidation_report_ref: ObjectRef | None = None
    incomplete_work_refs: tuple[ObjectRef, ...] = ()
    revalidation_ready_work_refs: tuple[ObjectRef, ...] = ()
    job_status: Literal[JobStatus.RUNNING] = JobStatus.RUNNING
    job_version: int = Field(ge=0)
    result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.predecessor_interaction_ref,
            "user-checkpoint-interaction",
            "v2",
            "predecessor_interaction_ref",
        )
        _require_ref(
            self.decision_commit_ref,
            "user-decision-commit",
            "v2",
            "decision_commit_ref",
        )
        if (
            self.interaction.predecessor_interaction_ref != self.predecessor_interaction_ref
            or self.interaction.state is not UserCheckpointInteractionStateV2.RESUMED
            or self.interaction.decision_commit_ref != self.decision_commit_ref
            or self.interaction.job_status is not JobStatus.RUNNING
        ):
            raise ValueError("resume interaction binding is stale")
        if self.application_ref is not None:
            _require_ref(
                self.application_ref,
                "user-plan-application",
                "v2",
                "application_ref",
            )
        if self.revalidation_report_ref is not None:
            _require_ref(
                self.revalidation_report_ref,
                "directed-revalidation-report",
                "v2",
                "revalidation_report_ref",
            )
        _require_refs(
            self.incomplete_work_refs,
            "resolved-work-unit",
            "v2",
            "incomplete_work_refs",
        )
        _require_refs(
            self.revalidation_ready_work_refs,
            "revalidation-work-item",
            "v2",
            "revalidation_ready_work_refs",
        )
        if self.incomplete_work_refs and self.revalidation_ready_work_refs:
            raise ValueError("resume result cannot mix normal and revalidation work")
        if self.application_ref is None and (
            self.revalidation_report_ref is not None or self.revalidation_ready_work_refs
        ):
            raise ValueError("revalidation evidence requires an application")
        refs = tuple(
            ref
            for ref in (
                self.predecessor_interaction_ref,
                user_checkpoint_interaction_v2_ref(self.interaction),
                self.decision_commit_ref,
                self.application_ref,
                self.revalidation_report_ref,
                *self.incomplete_work_refs,
                *self.revalidation_ready_work_refs,
            )
            if ref is not None
        )
        _validate_audit(self.audit, refs, "checkpoint resume result")
        _validate_identity(
            self.result_id,
            self.result_sha256,
            "user-checkpoint-resume-result",
            user_checkpoint_resume_result_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        predecessor_interaction_ref: ObjectRef,
        interaction: UserCheckpointInteractionV2,
        decision_commit_ref: ObjectRef,
        application_ref: ObjectRef | None,
        revalidation_report_ref: ObjectRef | None,
        incomplete_work_refs: tuple[ObjectRef, ...],
        revalidation_ready_work_refs: tuple[ObjectRef, ...],
        job_version: int,
        audit: ContractAudit,
    ) -> UserCheckpointResumeResultV2:
        refs = tuple(
            ref
            for ref in (
                predecessor_interaction_ref,
                user_checkpoint_interaction_v2_ref(interaction),
                decision_commit_ref,
                application_ref,
                revalidation_report_ref,
                *incomplete_work_refs,
                *revalidation_ready_work_refs,
            )
            if ref is not None
        )
        value = cls(
            result_id="user-checkpoint-resume-result://pending",
            predecessor_interaction_ref=predecessor_interaction_ref,
            interaction=interaction,
            decision_commit_ref=decision_commit_ref,
            application_ref=application_ref,
            revalidation_report_ref=revalidation_report_ref,
            incomplete_work_refs=_sorted_refs(incomplete_work_refs),
            revalidation_ready_work_refs=_sorted_refs(revalidation_ready_work_refs),
            job_version=job_version,
            result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _round_trip(
            _finalize(
                value,
                "result_id",
                "result_sha256",
                "user-checkpoint-resume-result",
                user_checkpoint_resume_result_v2_carried_sha256(value),
            )
        )


class UserCheckpointShowResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-checkpoint-show-result/v2"] = (
        "eval-factory/user-checkpoint-show-result/v2"
    )
    interaction: UserCheckpointInteractionV2
    request: UserApprovalRequest
    presentation: UserCheckpointPresentationV2

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        request_ref = user_approval_request_ref(self.request)
        presentation_ref = user_checkpoint_presentation_v2_ref(self.presentation)
        if (
            self.interaction.request_ref != request_ref
            or self.interaction.presentation_ref != presentation_ref
            or self.interaction.job_id != self.presentation.job_id
            or self.interaction.checkpoint is not self.request.checkpoint
            or self.presentation.request_ref != request_ref
            or self.presentation.checkpoint is not self.request.checkpoint
        ):
            raise ValueError("checkpoint show result bindings are stale")
        return self


class UserCheckpointInteractionPageV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-checkpoint-interaction-page/v2"] = (
        "eval-factory/user-checkpoint-interaction-page/v2"
    )
    job_id: Identifier
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=10_000)
    interactions: tuple[UserCheckpointInteractionV2, ...]

    @model_validator(mode="after")
    def validate_page(self) -> Self:
        if self.offset > self.total:
            raise ValueError("interaction page offset exceeds total")
        if len(self.interactions) > self.limit or self.offset + len(self.interactions) > self.total:
            raise ValueError("interaction page is not bounded")
        if any(value.job_id != self.job_id for value in self.interactions):
            raise ValueError("interaction page crosses Job identity")
        keys = tuple((value.chain_id, value.interaction_version) for value in self.interactions)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("interaction page must be sorted and unique")
        return self


def user_checkpoint_interaction_policy_v2_carried_sha256(
    value: UserCheckpointInteractionPolicyV2,
) -> str:
    return _model_payload_sha256(value, exclude={"policy_id", "policy_sha256", "audit"})


def user_checkpoint_presentation_v2_carried_sha256(
    value: UserCheckpointPresentationV2,
) -> str:
    body_ref = _presentation_body_ref(
        value.checkpoint,
        label_plan=value.label_plan,
        task_rewrite_preview=value.task_rewrite_preview,
        environment_strategy=value.environment_strategy,
        final_dataset_review_preview=value.final_dataset_review_preview,
    )
    return _payload_sha256(
        {
            "schema_version": value.schema_version,
            "job_id": value.job_id,
            "request_ref": _ref_payload(value.request_ref),
            "checkpoint": value.checkpoint.value,
            "body_ref": _ref_payload(body_ref),
        }
    )


def user_checkpoint_interaction_v2_carried_sha256(
    value: UserCheckpointInteractionV2,
) -> str:
    return _model_payload_sha256(
        value,
        exclude={"interaction_id", "interaction_sha256", "audit"},
    )


def user_checkpoint_open_result_v2_carried_sha256(
    value: UserCheckpointOpenResultV2,
) -> str:
    return _payload_sha256(
        {
            "schema_version": value.schema_version,
            "job_id": value.job_id,
            "outcome": value.outcome.value,
            "interaction_ref": (
                _ref_payload(user_checkpoint_interaction_v2_ref(value.interaction))
                if value.interaction is not None
                else None
            ),
            "job_status": value.job_status.value,
            "job_version": value.job_version,
            "policy_ref": _ref_payload(value.policy_ref),
        }
    )


def user_checkpoint_decision_result_v2_carried_sha256(
    value: UserCheckpointDecisionResultV2,
) -> str:
    return _payload_sha256(
        {
            "schema_version": value.schema_version,
            "predecessor_interaction_ref": _ref_payload(value.predecessor_interaction_ref),
            "interaction_ref": _ref_payload(user_checkpoint_interaction_v2_ref(value.interaction)),
            "next_interaction_ref": (
                _ref_payload(user_checkpoint_interaction_v2_ref(value.next_interaction))
                if value.next_interaction is not None
                else None
            ),
            "decision_commit_ref": _ref_payload(user_decision_commit_result_v2_ref(value.decision_commit)),
            "resume_disposition": value.resume_disposition.value,
            "job_status": value.job_status.value,
            "job_version": value.job_version,
        }
    )


def user_checkpoint_resume_result_v2_carried_sha256(
    value: UserCheckpointResumeResultV2,
) -> str:
    return _payload_sha256(
        {
            "schema_version": value.schema_version,
            "predecessor_interaction_ref": _ref_payload(value.predecessor_interaction_ref),
            "interaction_ref": _ref_payload(user_checkpoint_interaction_v2_ref(value.interaction)),
            "decision_commit_ref": _ref_payload(value.decision_commit_ref),
            "application_ref": (
                _ref_payload(value.application_ref) if value.application_ref is not None else None
            ),
            "revalidation_report_ref": (
                _ref_payload(value.revalidation_report_ref)
                if value.revalidation_report_ref is not None
                else None
            ),
            "incomplete_work_refs": [_ref_payload(ref) for ref in value.incomplete_work_refs],
            "revalidation_ready_work_refs": [_ref_payload(ref) for ref in value.revalidation_ready_work_refs],
            "job_status": value.job_status.value,
            "job_version": value.job_version,
        }
    )


def user_checkpoint_interaction_policy_v2_ref(
    value: UserCheckpointInteractionPolicyV2,
) -> ObjectRef:
    validate_user_checkpoint_interaction_policy_v2_identity(value)
    return ObjectRef(
        object_type="user-checkpoint-interaction-policy",
        object_id=value.policy_id,
        object_version="v2",
        object_sha256=value.policy_sha256,
    )


def user_checkpoint_presentation_v2_ref(
    value: UserCheckpointPresentationV2,
) -> ObjectRef:
    validate_user_checkpoint_presentation_v2_identity(value)
    return ObjectRef(
        object_type="user-checkpoint-presentation",
        object_id=value.presentation_id,
        object_version="v2",
        object_sha256=value.presentation_sha256,
    )


def user_checkpoint_interaction_v2_ref(
    value: UserCheckpointInteractionV2,
) -> ObjectRef:
    validate_user_checkpoint_interaction_v2_identity(value)
    return ObjectRef(
        object_type="user-checkpoint-interaction",
        object_id=value.interaction_id,
        object_version="v2",
        object_sha256=value.interaction_sha256,
    )


def user_checkpoint_open_result_v2_ref(
    value: UserCheckpointOpenResultV2,
) -> ObjectRef:
    validate_user_checkpoint_open_result_v2_identity(value)
    return ObjectRef(
        object_type="user-checkpoint-open-result",
        object_id=value.result_id,
        object_version="v2",
        object_sha256=value.result_sha256,
    )


def user_checkpoint_decision_result_v2_ref(
    value: UserCheckpointDecisionResultV2,
) -> ObjectRef:
    validate_user_checkpoint_decision_result_v2_identity(value)
    return ObjectRef(
        object_type="user-checkpoint-decision-result",
        object_id=value.result_id,
        object_version="v2",
        object_sha256=value.result_sha256,
    )


def user_checkpoint_resume_result_v2_ref(
    value: UserCheckpointResumeResultV2,
) -> ObjectRef:
    validate_user_checkpoint_resume_result_v2_identity(value)
    return ObjectRef(
        object_type="user-checkpoint-resume-result",
        object_id=value.result_id,
        object_version="v2",
        object_sha256=value.result_sha256,
    )


def validate_user_checkpoint_interaction_policy_v2_identity(
    value: UserCheckpointInteractionPolicyV2,
) -> None:
    _validate_identity(
        value.policy_id,
        value.policy_sha256,
        "user-checkpoint-interaction-policy",
        user_checkpoint_interaction_policy_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_user_checkpoint_presentation_v2_identity(
    value: UserCheckpointPresentationV2,
) -> None:
    _validate_identity(
        value.presentation_id,
        value.presentation_sha256,
        "user-checkpoint-presentation",
        user_checkpoint_presentation_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_user_checkpoint_interaction_v2_identity(
    value: UserCheckpointInteractionV2,
) -> None:
    _validate_identity(
        value.interaction_id,
        value.interaction_sha256,
        "user-checkpoint-interaction",
        user_checkpoint_interaction_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_user_checkpoint_open_result_v2_identity(
    value: UserCheckpointOpenResultV2,
) -> None:
    _validate_identity(
        value.result_id,
        value.result_sha256,
        "user-checkpoint-open-result",
        user_checkpoint_open_result_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_user_checkpoint_decision_result_v2_identity(
    value: UserCheckpointDecisionResultV2,
) -> None:
    _validate_identity(
        value.result_id,
        value.result_sha256,
        "user-checkpoint-decision-result",
        user_checkpoint_decision_result_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_user_checkpoint_resume_result_v2_identity(
    value: UserCheckpointResumeResultV2,
) -> None:
    _validate_identity(
        value.result_id,
        value.result_sha256,
        "user-checkpoint-resume-result",
        user_checkpoint_resume_result_v2_carried_sha256(value),
        allow_pending=False,
    )


def _presentation_body_ref(
    checkpoint: ApprovalCheckpoint,
    *,
    label_plan: LabelPlan | None,
    task_rewrite_preview: TaskRewritePlanPreviewV2 | None,
    environment_strategy: EnvironmentStrategy | None,
    final_dataset_review_preview: FinalDatasetReviewPreviewV2 | None,
) -> ObjectRef:
    bodies = (
        label_plan,
        task_rewrite_preview,
        environment_strategy,
        final_dataset_review_preview,
    )
    if sum(value is not None for value in bodies) != 1:
        raise ValueError("checkpoint presentation requires exactly one body")
    if checkpoint is ApprovalCheckpoint.LABEL_PLAN and label_plan is not None:
        return label_plan_ref(label_plan)
    if checkpoint is ApprovalCheckpoint.TASK_REWRITE_PLAN and task_rewrite_preview is not None:
        return task_rewrite_plan_preview_ref(task_rewrite_preview)
    if checkpoint is ApprovalCheckpoint.ENVIRONMENT_STRATEGY and environment_strategy is not None:
        return environment_strategy_ref(environment_strategy)
    if checkpoint is ApprovalCheckpoint.FINAL_DATASET_REVIEW and final_dataset_review_preview is not None:
        return final_dataset_review_preview_v2_ref(final_dataset_review_preview)
    raise ValueError("presentation body does not match checkpoint")


def _interaction_refs(
    value: UserCheckpointInteractionV2,
) -> tuple[ObjectRef, ...]:
    return tuple(
        ref
        for ref in (
            value.predecessor_interaction_ref,
            value.request_compilation_ref,
            value.request_ref,
            value.presentation_ref,
            value.source_context_ref,
            value.decision_commit_ref,
            value.application_ref,
            value.policy_ref,
        )
        if ref is not None
    )


def _model_payload_sha256(
    value: ContractModelV2,
    *,
    exclude: set[str],
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="python",
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


def _ref_payload(value: ObjectRef) -> dict[str, str]:
    return {
        "object_type": value.object_type,
        "object_id": value.object_id,
        "object_version": value.object_version,
        "object_sha256": value.object_sha256,
    }


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


def _round_trip[ModelT: ContractModelV2](value: ModelT) -> ModelT:
    return type(value).model_validate(value.model_dump(mode="python"))


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


def _require_ref(
    value: ObjectRef | None,
    object_type: str,
    object_version: str,
    field_name: str,
) -> None:
    if value is None or value.object_type != object_type or value.object_version != object_version:
        raise ValueError(f"{field_name} must reference {object_type} {object_version}")


def _require_refs(
    values: tuple[ObjectRef, ...],
    object_type: str,
    object_version: str,
    field_name: str,
) -> None:
    if values != _sorted_refs(values) or len(values) != len(set(values)):
        raise ValueError(f"{field_name} must be sorted and unique")
    for value in values:
        _require_ref(value, object_type, object_version, field_name)


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(sorted(set(values), key=_ref_key))


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return audit.model_copy(update={"input_refs": _sorted_refs(refs)})


def _validate_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != _sorted_refs(refs):
        raise ValueError(f"{label} audit refs are incomplete")


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


__all__ = [
    "CHECKPOINT_INTERACTION_POLICY_VERSION",
    "UserCheckpointDecisionResultV2",
    "UserCheckpointDecisionSubmissionV2",
    "UserCheckpointInteractionPageV2",
    "UserCheckpointInteractionPolicyV2",
    "UserCheckpointInteractionStateV2",
    "UserCheckpointInteractionV2",
    "UserCheckpointOpenOutcomeV2",
    "UserCheckpointOpenResultV2",
    "UserCheckpointPresentationV2",
    "UserCheckpointResumeDispositionV2",
    "UserCheckpointResumeResultV2",
    "UserCheckpointShowResultV2",
    "user_checkpoint_decision_result_v2_ref",
    "user_checkpoint_interaction_policy_v2_ref",
    "user_checkpoint_interaction_v2_ref",
    "user_checkpoint_open_result_v2_ref",
    "user_checkpoint_presentation_v2_ref",
    "user_checkpoint_resume_result_v2_ref",
    "validate_user_checkpoint_decision_result_v2_identity",
    "validate_user_checkpoint_interaction_policy_v2_identity",
    "validate_user_checkpoint_interaction_v2_identity",
    "validate_user_checkpoint_open_result_v2_identity",
    "validate_user_checkpoint_presentation_v2_identity",
    "validate_user_checkpoint_resume_result_v2_identity",
]
