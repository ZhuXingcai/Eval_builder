from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.approval import (
    DECISIONS_BY_CHECKPOINT,
    ApprovalCheckpoint,
    EnvironmentStrategy,
    FinalReviewScope,
    LabelPlan,
    UserApprovalPolicy,
    UserApprovalRequest,
)
from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
    VersionBinding,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2

USER_APPROVAL_REQUEST_POLICY_VERSION: Literal["user-approval-request/r7-04-v1"] = (
    "user-approval-request/r7-04-v1"
)


class UserApprovalRequestCompilationOutcomeV2(StrEnum):
    REQUESTED = "REQUESTED"
    DISABLED = "DISABLED"
    NOT_REQUIRED = "NOT_REQUIRED"


class ApprovalRequestGenerationPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/approval-request-generation-policy/v2"] = (
        "eval-factory/approval-request-generation-policy/v2"
    )
    generation_policy_id: Identifier
    max_requests_per_checkpoint: int = Field(ge=1, le=100_000)
    max_subject_refs_per_request: int = Field(ge=1, le=100_000)
    max_preview_refs_per_request: int = Field(ge=1, le=100_000)
    max_examples_per_plan: int = Field(ge=1, le=100_000)
    max_preview_characters_per_request: int = Field(
        ge=1,
        le=100_000_000,
    )
    max_preview_characters_per_checkpoint: int = Field(
        ge=1,
        le=1_000_000_000,
    )
    max_final_sample_refs: int = Field(ge=1, le=1_000_000)
    policy_version: Literal["user-approval-request/r7-04-v1"] = USER_APPROVAL_REQUEST_POLICY_VERSION
    policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if self.max_preview_characters_per_request > self.max_preview_characters_per_checkpoint:
            raise ValueError("per-request preview character limit cannot exceed checkpoint limit")
        _validate_audit(
            self.audit,
            (),
            "approval request generation policy",
        )
        _validate_identity(
            object_id=self.generation_policy_id,
            object_sha256=self.policy_sha256,
            expected_prefix="approval-request-generation-policy",
            observed=approval_request_generation_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        max_requests_per_checkpoint: int,
        max_subject_refs_per_request: int,
        max_preview_refs_per_request: int,
        max_examples_per_plan: int,
        max_preview_characters_per_request: int,
        max_preview_characters_per_checkpoint: int,
        max_final_sample_refs: int,
        audit: ContractAudit,
    ) -> ApprovalRequestGenerationPolicyV2:
        value = cls(
            generation_policy_id=("approval-request-generation-policy://pending"),
            max_requests_per_checkpoint=max_requests_per_checkpoint,
            max_subject_refs_per_request=max_subject_refs_per_request,
            max_preview_refs_per_request=max_preview_refs_per_request,
            max_examples_per_plan=max_examples_per_plan,
            max_preview_characters_per_request=(max_preview_characters_per_request),
            max_preview_characters_per_checkpoint=(max_preview_characters_per_checkpoint),
            max_final_sample_refs=max_final_sample_refs,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, ()),
        )
        return _finalize(
            value,
            id_field="generation_policy_id",
            hash_field="policy_sha256",
            prefix="approval-request-generation-policy",
            digest=(approval_request_generation_policy_v2_carried_sha256(value)),
        )


class FinalDatasetReviewPreviewV2(ContractModelV2):
    schema_version: Literal["eval-factory/final-dataset-review-preview/v2"] = (
        "eval-factory/final-dataset-review-preview/v2"
    )
    preview_id: Identifier
    scope: FinalReviewScope
    subject_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    prompt_projection_refs: tuple[ObjectRef, ...] = ()
    selected_item_projection_refs: tuple[ObjectRef, ...] = ()
    dataset_projection_ref: ObjectRef | None = None
    quality_summary_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    open_low_severity_finding_refs: tuple[ObjectRef, ...] = ()
    sample_navigation_refs: tuple[ObjectRef, ...] = ()
    subject_count: int = Field(ge=1)
    prompt_count: int = Field(ge=0)
    selected_item_count: int = Field(ge=0)
    open_low_severity_finding_count: int = Field(ge=0)
    automated_hard_gates_passed: Literal[True] = True
    projection_policy_version: str = Field(min_length=1, max_length=128)
    preview_sha256: Sha256
    audit: ContractAudit

    @field_validator("scope", mode="before")
    @classmethod
    def parse_scope(cls, value: object) -> FinalReviewScope:
        return _parse_enum(value, FinalReviewScope, "scope")

    @model_validator(mode="after")
    def validate_preview(self) -> Self:
        if self.scope is FinalReviewScope.NONE:
            raise ValueError("final review preview scope cannot be NONE")
        for label, refs, expected_type in (
            (
                "final review prompt projection refs",
                self.prompt_projection_refs,
                "user-prompt-projection",
            ),
            (
                "final review selected Item projection refs",
                self.selected_item_projection_refs,
                "user-item-projection",
            ),
            (
                "final review low-severity finding refs",
                self.open_low_severity_finding_refs,
                "low-severity-finding-projection",
            ),
            (
                "final review sample navigation refs",
                self.sample_navigation_refs,
                "approval-navigation-sample",
            ),
        ):
            _require_sorted_unique_refs(label, refs)
            for ref in refs:
                _require_ref(ref, expected_type, "v2", label)
        _require_sorted_unique_refs(
            "final review subject refs",
            self.subject_refs,
        )
        for ref in self.subject_refs:
            _require_ref(
                ref,
                "evaluation-item",
                "v2",
                "final review subject refs",
            )
        _require_sorted_unique_refs(
            "final review quality summary refs",
            self.quality_summary_refs,
        )
        for ref in self.quality_summary_refs:
            if ref.object_type not in {
                "quality-report",
                "item-quality-report-revision",
                "batch-quality-report",
            } or ref.object_version not in {"v1", "v2"}:
                raise ValueError("quality_summary_refs contain an invalid reference")
        if self.dataset_projection_ref is not None:
            _require_ref(
                self.dataset_projection_ref,
                "user-dataset-projection",
                "v2",
                "dataset_projection_ref",
            )
        if self.scope is FinalReviewScope.PROMPTS:
            if (
                not self.prompt_projection_refs
                or self.selected_item_projection_refs
                or self.dataset_projection_ref is not None
            ):
                raise ValueError("PROMPTS final review requires prompt projections only")
        elif self.scope is FinalReviewScope.SELECTED_ITEMS:
            if not self.selected_item_projection_refs or self.dataset_projection_ref is not None:
                raise ValueError(
                    "SELECTED_ITEMS final review requires Item projections and no dataset projection"
                )
        elif self.dataset_projection_ref is None:
            raise ValueError("FULL_DATASET final review requires one dataset projection")
        expected_counts = (
            len(self.subject_refs),
            len(self.prompt_projection_refs),
            len(self.selected_item_projection_refs),
            len(self.open_low_severity_finding_refs),
        )
        observed_counts = (
            self.subject_count,
            self.prompt_count,
            self.selected_item_count,
            self.open_low_severity_finding_count,
        )
        if observed_counts != expected_counts:
            raise ValueError("final review preview counts are not exact")
        _validate_audit(
            self.audit,
            _final_dataset_review_refs(self),
            "final dataset review preview",
        )
        _validate_identity(
            object_id=self.preview_id,
            object_sha256=self.preview_sha256,
            expected_prefix="final-dataset-review-preview",
            observed=final_dataset_review_preview_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        scope: FinalReviewScope,
        subject_refs: tuple[ObjectRef, ...],
        prompt_projection_refs: tuple[ObjectRef, ...],
        selected_item_projection_refs: tuple[ObjectRef, ...],
        dataset_projection_ref: ObjectRef | None,
        quality_summary_refs: tuple[ObjectRef, ...],
        open_low_severity_finding_refs: tuple[ObjectRef, ...],
        sample_navigation_refs: tuple[ObjectRef, ...],
        projection_policy_version: str,
        audit: ContractAudit,
    ) -> FinalDatasetReviewPreviewV2:
        subjects = _sorted_refs(subject_refs)
        prompts = _sorted_refs(prompt_projection_refs)
        selected_items = _sorted_refs(selected_item_projection_refs)
        quality = _sorted_refs(quality_summary_refs)
        findings = _sorted_refs(open_low_severity_finding_refs)
        samples = _sorted_refs(sample_navigation_refs)
        refs = (
            *subjects,
            *prompts,
            *selected_items,
            *((dataset_projection_ref,) if dataset_projection_ref else ()),
            *quality,
            *findings,
            *samples,
        )
        value = cls(
            preview_id="final-dataset-review-preview://pending",
            scope=scope,
            subject_refs=subjects,
            prompt_projection_refs=prompts,
            selected_item_projection_refs=selected_items,
            dataset_projection_ref=dataset_projection_ref,
            quality_summary_refs=quality,
            open_low_severity_finding_refs=findings,
            sample_navigation_refs=samples,
            subject_count=len(subjects),
            prompt_count=len(prompts),
            selected_item_count=len(selected_items),
            open_low_severity_finding_count=len(findings),
            projection_policy_version=projection_policy_version,
            preview_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            id_field="preview_id",
            hash_field="preview_sha256",
            prefix="final-dataset-review-preview",
            digest=final_dataset_review_preview_v2_carried_sha256(value),
        )


class UserApprovalRequestCompilationResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/user-approval-request-compilation-result/v2"] = (
        "eval-factory/user-approval-request-compilation-result/v2"
    )
    result_id: Identifier
    job_id: Identifier
    dataset_job_spec_ref: ObjectRef
    approval_policy_ref: ObjectRef
    generation_policy_ref: ObjectRef
    checkpoint: ApprovalCheckpoint
    outcome: UserApprovalRequestCompilationOutcomeV2
    requests: tuple[UserApprovalRequest, ...] = ()
    request_refs: tuple[ObjectRef, ...] = ()
    request_preview_character_counts: tuple[int, ...] = ()
    request_count: int = Field(ge=0)
    subject_ref_count: int = Field(ge=0)
    preview_ref_count: int = Field(ge=0)
    preview_character_count: int = Field(ge=0)
    policy_version: Literal["user-approval-request/r7-04-v1"] = USER_APPROVAL_REQUEST_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("checkpoint", mode="before")
    @classmethod
    def parse_checkpoint(cls, value: object) -> ApprovalCheckpoint:
        return _parse_enum(value, ApprovalCheckpoint, "checkpoint")

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(
        cls,
        value: object,
    ) -> UserApprovalRequestCompilationOutcomeV2:
        return _parse_enum(
            value,
            UserApprovalRequestCompilationOutcomeV2,
            "outcome",
        )

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.dataset_job_spec_ref,
            "dataset-job-spec",
            "v2",
            "dataset_job_spec_ref",
        )
        _require_ref(
            self.approval_policy_ref,
            "user-approval-policy",
            "v2",
            "approval_policy_ref",
        )
        _require_ref(
            self.generation_policy_ref,
            "approval-request-generation-policy",
            "v2",
            "generation_policy_ref",
        )
        expected_refs = tuple(user_approval_request_ref(request) for request in self.requests)
        if self.request_refs != expected_refs:
            raise ValueError("request refs do not match nested approval requests")
        _require_sorted_unique_refs(
            "compiled approval request refs",
            self.request_refs,
        )
        if len(self.request_preview_character_counts) != len(self.requests):
            raise ValueError("request preview character counts must align with requests")
        if any(value < 0 for value in self.request_preview_character_counts):
            raise ValueError("request preview character counts must be non-negative")
        for request in self.requests:
            validate_user_approval_request_identity(request)
            if (
                request.checkpoint is not self.checkpoint
                or request.approval_policy_ref != self.approval_policy_ref
            ):
                raise ValueError("compiled approval request ownership is mismatched")
        if self.requests and len({request.requested_by for request in self.requests}) != 1:
            raise ValueError("compiled approval requests require one requesting user")
        expected_counts = (
            len(self.requests),
            sum(len(request.subject_refs) for request in self.requests),
            sum(len(request.preview_refs) for request in self.requests),
            sum(self.request_preview_character_counts),
        )
        observed_counts = (
            self.request_count,
            self.subject_ref_count,
            self.preview_ref_count,
            self.preview_character_count,
        )
        if observed_counts != expected_counts:
            raise ValueError("approval request compilation counts are not exact")
        if self.outcome is UserApprovalRequestCompilationOutcomeV2.REQUESTED:
            if not self.requests:
                raise ValueError("REQUESTED approval compilation requires requests")
        elif self.requests or any(observed_counts):
            raise ValueError("empty approval compilation outcome cannot carry requests")
        if (
            self.outcome is UserApprovalRequestCompilationOutcomeV2.NOT_REQUIRED
            and self.checkpoint is not ApprovalCheckpoint.ENVIRONMENT_STRATEGY
        ):
            raise ValueError("NOT_REQUIRED is valid only for ENVIRONMENT_STRATEGY")
        _validate_audit(
            self.audit,
            (
                self.dataset_job_spec_ref,
                self.approval_policy_ref,
                self.generation_policy_ref,
                *self.request_refs,
            ),
            "user approval request compilation result",
        )
        _validate_identity(
            object_id=self.result_id,
            object_sha256=self.result_sha256,
            expected_prefix="user-approval-request-compilation-result",
            observed=(user_approval_request_compilation_result_v2_carried_sha256(self)),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        job_id: str,
        dataset_job_spec_ref: ObjectRef,
        approval_policy_ref: ObjectRef,
        generation_policy_ref: ObjectRef,
        checkpoint: ApprovalCheckpoint,
        outcome: UserApprovalRequestCompilationOutcomeV2,
        requests: tuple[UserApprovalRequest, ...],
        request_preview_character_counts: tuple[int, ...],
        audit: ContractAudit,
    ) -> UserApprovalRequestCompilationResultV2:
        pairs = sorted(
            zip(
                requests,
                request_preview_character_counts,
                strict=True,
            ),
            key=lambda pair: _ref_key(user_approval_request_ref(pair[0])),
        )
        ordered_requests = tuple(pair[0] for pair in pairs)
        character_counts = tuple(pair[1] for pair in pairs)
        request_refs = tuple(user_approval_request_ref(request) for request in ordered_requests)
        refs = (
            dataset_job_spec_ref,
            approval_policy_ref,
            generation_policy_ref,
            *request_refs,
        )
        value = cls(
            result_id=("user-approval-request-compilation-result://pending"),
            job_id=job_id,
            dataset_job_spec_ref=dataset_job_spec_ref,
            approval_policy_ref=approval_policy_ref,
            generation_policy_ref=generation_policy_ref,
            checkpoint=checkpoint,
            outcome=outcome,
            requests=ordered_requests,
            request_refs=request_refs,
            request_preview_character_counts=character_counts,
            request_count=len(ordered_requests),
            subject_ref_count=sum(len(request.subject_refs) for request in ordered_requests),
            preview_ref_count=sum(len(request.preview_refs) for request in ordered_requests),
            preview_character_count=sum(character_counts),
            result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            id_field="result_id",
            hash_field="result_sha256",
            prefix="user-approval-request-compilation-result",
            digest=(user_approval_request_compilation_result_v2_carried_sha256(value)),
        )


def approval_request_generation_policy_v2_carried_sha256(
    value: ApprovalRequestGenerationPolicyV2,
) -> str:
    return _carried_sha256(
        value,
        exclude={
            "generation_policy_id",
            "policy_sha256",
            "audit",
        },
    )


def approval_request_generation_policy_v2_ref(
    value: ApprovalRequestGenerationPolicyV2,
) -> ObjectRef:
    validate_approval_request_generation_policy_v2_identity(value)
    return ObjectRef(
        object_type="approval-request-generation-policy",
        object_id=value.generation_policy_id,
        object_version="v2",
        object_sha256=value.policy_sha256,
    )


def final_dataset_review_preview_v2_carried_sha256(
    value: FinalDatasetReviewPreviewV2,
) -> str:
    return _carried_sha256(
        value,
        exclude={"preview_id", "preview_sha256", "audit"},
    )


def final_dataset_review_preview_v2_ref(
    value: FinalDatasetReviewPreviewV2,
) -> ObjectRef:
    validate_final_dataset_review_preview_v2_identity(value)
    return ObjectRef(
        object_type="final-dataset-review-preview",
        object_id=value.preview_id,
        object_version="v2",
        object_sha256=value.preview_sha256,
    )


def user_approval_request_compilation_result_v2_carried_sha256(
    value: UserApprovalRequestCompilationResultV2,
) -> str:
    return _payload_sha256(
        {
            "job_id": value.job_id,
            "dataset_job_spec_ref": _ref_payload(value.dataset_job_spec_ref),
            "approval_policy_ref": _ref_payload(value.approval_policy_ref),
            "generation_policy_ref": _ref_payload(value.generation_policy_ref),
            "checkpoint": value.checkpoint.value,
            "outcome": value.outcome.value,
            "request_refs": [_ref_payload(ref) for ref in value.request_refs],
            "request_preview_character_counts": list(value.request_preview_character_counts),
            "request_count": value.request_count,
            "subject_ref_count": value.subject_ref_count,
            "preview_ref_count": value.preview_ref_count,
            "preview_character_count": value.preview_character_count,
            "policy_version": value.policy_version,
        }
    )


def user_approval_request_compilation_result_v2_ref(
    value: UserApprovalRequestCompilationResultV2,
) -> ObjectRef:
    validate_user_approval_request_compilation_result_v2_identity(value)
    return ObjectRef(
        object_type="user-approval-request-compilation-result",
        object_id=value.result_id,
        object_version="v2",
        object_sha256=value.result_sha256,
    )


def user_approval_policy_carried_sha256(
    value: UserApprovalPolicy,
) -> str:
    payload = value.model_dump(
        mode="json",
        exclude={"policy_id", "audit"},
        exclude_none=False,
    )
    payload["enabled_checkpoints"] = sorted(checkpoint.value for checkpoint in value.enabled_checkpoints)
    return _payload_sha256(payload)


def user_approval_policy_ref(value: UserApprovalPolicy) -> ObjectRef:
    validate_user_approval_policy_identity(value)
    return ObjectRef(
        object_type="user-approval-policy",
        object_id=value.policy_id,
        object_version="v2",
        object_sha256=user_approval_policy_carried_sha256(value),
    )


def label_plan_carried_sha256(value: LabelPlan) -> str:
    payload = value.model_dump(
        mode="json",
        exclude={"label_plan_id", "audit"},
        exclude_none=False,
    )
    payload["deterministic_clause_refs"] = [
        _ref_payload(ref) for ref in _sorted_refs(value.deterministic_clause_refs)
    ]
    payload["semantic_clause_refs"] = [_ref_payload(ref) for ref in _sorted_refs(value.semantic_clause_refs)]
    examples = []
    for example in sorted(value.examples, key=lambda item: item.example_id):
        example_payload = example.model_dump(mode="json", exclude_none=False)
        example_payload["evidence_refs"] = [
            evidence.model_dump(mode="json", exclude_none=False)
            for evidence in sorted(
                example.evidence_refs,
                key=lambda item: item.evidence_ref_id,
            )
        ]
        examples.append(example_payload)
    payload["examples"] = examples
    payload["abstain_rules"] = sorted(value.abstain_rules)
    payload["expected_model_path"] = list(value.expected_model_path)
    payload["blind_spots"] = sorted(value.blind_spots)
    return _payload_sha256(payload)


def label_plan_ref(value: LabelPlan) -> ObjectRef:
    _require_ref(
        value.label_spec_ref,
        "label-spec",
        "v2",
        "LabelPlan label_spec_ref",
    )
    _validate_source_audit(
        value.audit,
        (value.label_spec_ref,),
        "LabelPlan",
    )
    observed = label_plan_carried_sha256(value)
    _require_scaffold_id(
        value.label_plan_id,
        "label-plan",
        observed,
    )
    return ObjectRef(
        object_type="label-plan",
        object_id=value.label_plan_id,
        object_version="v2",
        object_sha256=observed,
    )


def environment_strategy_carried_sha256(
    value: EnvironmentStrategy,
) -> str:
    requirements = []
    for requirement in sorted(
        value.requirements,
        key=lambda item: item.requirement_id,
    ):
        payload = requirement.model_dump(mode="json", exclude_none=False)
        payload["affected_subject_refs"] = [
            _ref_payload(ref) for ref in _sorted_refs(requirement.affected_subject_refs)
        ]
        payload["evidence_refs"] = [
            evidence.model_dump(mode="json", exclude_none=False)
            for evidence in sorted(
                requirement.evidence_refs,
                key=lambda item: item.evidence_ref_id,
            )
        ]
        payload["alternatives"] = [
            alternative.model_dump(mode="json", exclude_none=False)
            for alternative in sorted(
                requirement.alternatives,
                key=lambda item: item.strategy.value,
            )
        ]
        requirements.append(payload)
    return _payload_sha256(
        {
            "schema_version": value.schema_version,
            "requirements": requirements,
            "query_packaging_options": sorted(item.value for item in value.query_packaging_options),
            "recommended_strategy": (
                value.recommended_strategy.value if value.recommended_strategy else None
            ),
            "recommendation_reason": value.recommendation_reason,
            "credential_fabrication_forbidden": (value.credential_fabrication_forbidden),
        }
    )


def environment_strategy_ref(value: EnvironmentStrategy) -> ObjectRef:
    affected_subject_refs = tuple(
        ref for requirement in value.requirements for ref in requirement.affected_subject_refs
    )
    _validate_source_audit(
        value.audit,
        affected_subject_refs,
        "EnvironmentStrategy",
    )
    observed = environment_strategy_carried_sha256(value)
    _require_scaffold_id(
        value.environment_strategy_id,
        "environment-strategy",
        observed,
    )
    return ObjectRef(
        object_type="environment-strategy",
        object_id=value.environment_strategy_id,
        object_version="v2",
        object_sha256=observed,
    )


def user_approval_request_carried_sha256(
    value: UserApprovalRequest,
) -> str:
    return _payload_sha256(
        {
            "schema_version": value.schema_version,
            "checkpoint": value.checkpoint.value,
            "requested_by": value.requested_by,
            "approval_policy_ref": _ref_payload(value.approval_policy_ref),
            "subject_refs": [_ref_payload(ref) for ref in _sorted_refs(value.subject_refs)],
            "plan_ref": (_ref_payload(value.plan_ref) if value.plan_ref is not None else None),
            "projection_ref": _ref_payload(value.projection_ref),
            "preview_refs": [_ref_payload(ref) for ref in _sorted_refs(value.preview_refs)],
            "available_decisions": sorted(item.value for item in value.available_decisions),
            "affected_stages": sorted(value.affected_stages),
            "idempotency_key": value.idempotency_key,
        }
    )


def user_approval_request_ref(value: UserApprovalRequest) -> ObjectRef:
    validate_user_approval_request_identity(value)
    return ObjectRef(
        object_type="user-approval-request",
        object_id=value.request_id,
        object_version="v2",
        object_sha256=user_approval_request_carried_sha256(value),
    )


def validate_approval_request_generation_policy_v2_identity(
    value: ApprovalRequestGenerationPolicyV2,
) -> None:
    _validate_audit(
        value.audit,
        (),
        "approval request generation policy",
    )
    _require_current_identity(
        object_id=value.generation_policy_id,
        object_sha256=value.policy_sha256,
        expected_prefix="approval-request-generation-policy",
        observed=approval_request_generation_policy_v2_carried_sha256(value),
    )


def validate_final_dataset_review_preview_v2_identity(
    value: FinalDatasetReviewPreviewV2,
) -> None:
    _validate_audit(
        value.audit,
        _final_dataset_review_refs(value),
        "final dataset review preview",
    )
    _require_current_identity(
        object_id=value.preview_id,
        object_sha256=value.preview_sha256,
        expected_prefix="final-dataset-review-preview",
        observed=final_dataset_review_preview_v2_carried_sha256(value),
    )


def validate_user_approval_request_compilation_result_v2_identity(
    value: UserApprovalRequestCompilationResultV2,
) -> None:
    _validate_audit(
        value.audit,
        (
            value.dataset_job_spec_ref,
            value.approval_policy_ref,
            value.generation_policy_ref,
            *value.request_refs,
        ),
        "user approval request compilation result",
    )
    _require_current_identity(
        object_id=value.result_id,
        object_sha256=value.result_sha256,
        expected_prefix="user-approval-request-compilation-result",
        observed=(user_approval_request_compilation_result_v2_carried_sha256(value)),
    )


def validate_user_approval_policy_identity(
    value: UserApprovalPolicy,
) -> None:
    if value.policy_version != USER_APPROVAL_REQUEST_POLICY_VERSION:
        raise ValueError("user approval policy version is stale")
    _validate_audit(value.audit, (), "user approval policy")
    observed = user_approval_policy_carried_sha256(value)
    _require_scaffold_id(value.policy_id, "user-approval-policy", observed)


def validate_user_approval_request_identity(
    value: UserApprovalRequest,
) -> None:
    _require_ref(
        value.approval_policy_ref,
        "user-approval-policy",
        "v2",
        "approval_policy_ref",
    )
    _require_sorted_unique_refs(
        "user approval request subject refs",
        value.subject_refs,
    )
    _require_safe_refs(value.subject_refs, "subject_refs")
    _require_ref(
        value.projection_ref,
        "user-approval-projection",
        "v2",
        "projection_ref",
    )
    _require_sorted_unique_refs(
        "user approval request preview refs",
        value.preview_refs,
    )
    _validate_request_ref_shapes(value)
    if value.available_decisions != DECISIONS_BY_CHECKPOINT[value.checkpoint]:
        raise ValueError(f"{value.checkpoint.value} available decisions are not exact")
    expected_stages = {
        ApprovalCheckpoint.LABEL_PLAN: ("label",),
        ApprovalCheckpoint.TASK_REWRITE_PLAN: ("task-authoring",),
        ApprovalCheckpoint.ENVIRONMENT_STRATEGY: (
            "attachment",
            "task-authoring",
        ),
        ApprovalCheckpoint.FINAL_DATASET_REVIEW: ("release",),
    }[value.checkpoint]
    if tuple(sorted(value.affected_stages)) != expected_stages:
        raise ValueError("user approval request affected stages are not exact")
    expected_audit_refs = (
        value.approval_policy_ref,
        *value.subject_refs,
        *((value.plan_ref,) if value.plan_ref is not None else ()),
        value.projection_ref,
        *value.preview_refs,
    )
    _validate_audit(
        value.audit,
        expected_audit_refs,
        "user approval request",
    )
    observed = user_approval_request_carried_sha256(value)
    _require_scaffold_id(
        value.request_id,
        "user-approval-request",
        observed,
    )


def _validate_request_ref_shapes(
    value: UserApprovalRequest,
) -> None:
    if value.checkpoint is ApprovalCheckpoint.LABEL_PLAN:
        if len(value.subject_refs) != 1:
            raise ValueError("LABEL_PLAN requires one label-spec subject")
        _require_ref(
            value.subject_refs[0],
            "label-spec",
            "v2",
            "LABEL_PLAN subject_refs",
        )
        if value.plan_ref is None:
            raise ValueError("LABEL_PLAN requires a label-plan")
        _require_ref(
            value.plan_ref,
            "label-plan",
            "v2",
            "LABEL_PLAN plan_ref",
        )
        if value.preview_refs != (value.plan_ref,):
            raise ValueError("LABEL_PLAN preview_refs must contain only its label-plan")
        return
    if value.checkpoint is ApprovalCheckpoint.TASK_REWRITE_PLAN:
        if len(value.subject_refs) != 1:
            raise ValueError("TASK_REWRITE_PLAN requires one selection-context subject")
        _require_ref(
            value.subject_refs[0],
            "selection-context",
            "v2",
            "TASK_REWRITE_PLAN subject_refs",
        )
        if value.plan_ref is None:
            raise ValueError("TASK_REWRITE_PLAN requires a versioned task rewrite plan")
        _require_ref(
            value.plan_ref,
            "task-rewrite-plan-version",
            "v2",
            "TASK_REWRITE_PLAN plan_ref",
        )
        if len(value.preview_refs) != 1:
            raise ValueError("TASK_REWRITE_PLAN requires one safe preview ref")
        _require_ref(
            value.preview_refs[0],
            "task-rewrite-plan-preview",
            "v2",
            "TASK_REWRITE_PLAN preview_refs",
        )
        return
    if value.checkpoint is ApprovalCheckpoint.ENVIRONMENT_STRATEGY:
        for ref in value.subject_refs:
            _require_ref(
                ref,
                "task-draft",
                "v2",
                "ENVIRONMENT_STRATEGY subject_refs",
            )
        if value.plan_ref is None:
            raise ValueError("ENVIRONMENT_STRATEGY requires an environment strategy")
        _require_ref(
            value.plan_ref,
            "environment-strategy",
            "v2",
            "ENVIRONMENT_STRATEGY plan_ref",
        )
        if value.preview_refs != (value.plan_ref,):
            raise ValueError("ENVIRONMENT_STRATEGY preview_refs must contain only its strategy")
        return
    for ref in value.subject_refs:
        _require_ref(
            ref,
            "evaluation-item",
            "v2",
            "FINAL_DATASET_REVIEW subject_refs",
        )
    if value.plan_ref is not None:
        raise ValueError("FINAL_DATASET_REVIEW cannot bind a plan")
    if len(value.preview_refs) != 1:
        raise ValueError("FINAL_DATASET_REVIEW requires one bounded final review preview")
    _require_ref(
        value.preview_refs[0],
        "final-dataset-review-preview",
        "v2",
        "FINAL_DATASET_REVIEW preview_refs",
    )


def _final_dataset_review_refs(
    value: FinalDatasetReviewPreviewV2,
) -> tuple[ObjectRef, ...]:
    return (
        *value.subject_refs,
        *value.prompt_projection_refs,
        *value.selected_item_projection_refs,
        *((value.dataset_projection_ref,) if value.dataset_projection_ref else ()),
        *value.quality_summary_refs,
        *value.open_low_severity_finding_refs,
        *value.sample_navigation_refs,
    )


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
    bindings = tuple(
        value for value in audit.governing_versions if value.component == "user-approval-request"
    )
    if len(bindings) != 1 or bindings[0].version != USER_APPROVAL_REQUEST_POLICY_VERSION:
        raise ValueError(f"{label} audit is missing the current approval policy")


def _validate_source_audit(
    audit: ContractAudit,
    expected_refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != _sorted_refs(expected_refs):
        raise ValueError(f"{label} audit input refs are not exact")


def _safe_audit(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    governing = (
        *(value for value in audit.governing_versions if value.component != "user-approval-request"),
        VersionBinding(
            component="user-approval-request",
            version=USER_APPROVAL_REQUEST_POLICY_VERSION,
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


def _require_scaffold_id(
    object_id: str,
    prefix: str,
    observed: str,
) -> None:
    if object_id != f"{prefix}://sha256/{observed}":
        raise ValueError(f"{prefix} identity is stale")


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
