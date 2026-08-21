from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import model_validator

from env_mock_agent.facade.contracts import (
    FacadeModel,
    FacadeObjectRef,
    Identifier,
    Sha256,
)

ATTACHMENT_SEMANTIC_REVIEW_POLICY_VERSION: Literal["semantic-review/r5-09-v1"] = "semantic-review/r5-09-v1"
ATTACHMENT_REPAIR_POLICY_VERSION: Literal["targeted-repair/r5-09-v1"] = "targeted-repair/r5-09-v1"


class AttachmentSemanticReviewRoundV2(StrEnum):
    COVERAGE_SOLVABILITY = "COVERAGE_SOLVABILITY"
    REALISM_CONSISTENCY = "REALISM_CONSISTENCY"
    LEAKAGE_EXECUTABILITY = "LEAKAGE_EXECUTABILITY"


class AttachmentSemanticReviewerRoleV2(StrEnum):
    COVERAGE_SOLVABILITY_REVIEWER = "COVERAGE_SOLVABILITY_REVIEWER"
    REALISM_CONSISTENCY_REVIEWER = "REALISM_CONSISTENCY_REVIEWER"
    LEAKAGE_EXECUTABILITY_REVIEWER = "LEAKAGE_EXECUTABILITY_REVIEWER"


ROUND_ROLES: dict[
    AttachmentSemanticReviewRoundV2,
    AttachmentSemanticReviewerRoleV2,
] = {
    AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY: (
        AttachmentSemanticReviewerRoleV2.COVERAGE_SOLVABILITY_REVIEWER
    ),
    AttachmentSemanticReviewRoundV2.REALISM_CONSISTENCY: (
        AttachmentSemanticReviewerRoleV2.REALISM_CONSISTENCY_REVIEWER
    ),
    AttachmentSemanticReviewRoundV2.LEAKAGE_EXECUTABILITY: (
        AttachmentSemanticReviewerRoleV2.LEAKAGE_EXECUTABILITY_REVIEWER
    ),
}


class AttachmentSemanticReviewSeverityV2(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class AttachmentSemanticFindingScopeV2(StrEnum):
    ITEM = "ITEM"
    ARTIFACT = "ARTIFACT"


class AttachmentSemanticReviewFindingCodeV2(StrEnum):
    REQUIRED_COVERAGE_MISSING = "REQUIRED_COVERAGE_MISSING"
    DEPENDENCY_NOT_SOLVABLE = "DEPENDENCY_NOT_SOLVABLE"
    PUBLIC_RUBRIC_UNREACHABLE = "PUBLIC_RUBRIC_UNREACHABLE"
    SAFE_EVIDENCE_INSUFFICIENT = "SAFE_EVIDENCE_INSUFFICIENT"
    CAPABILITY_CONTRACT_MISMATCH = "CAPABILITY_CONTRACT_MISMATCH"
    WORLD_FACT_INCONSISTENT = "WORLD_FACT_INCONSISTENT"
    SOURCE_EVIDENCE_INCONSISTENT = "SOURCE_EVIDENCE_INCONSISTENT"
    ARTIFACT_CONTENT_UNREALISTIC = "ARTIFACT_CONTENT_UNREALISTIC"
    CROSS_ARTIFACT_INCONSISTENT = "CROSS_ARTIFACT_INCONSISTENT"
    ARTIFACT_METADATA_INCONSISTENT = "ARTIFACT_METADATA_INCONSISTENT"
    ANSWER_BEARING_CONTENT = "ANSWER_BEARING_CONTENT"
    COMPLETED_DELIVERABLE_EXPOSED = "COMPLETED_DELIVERABLE_EXPOSED"
    PRIVATE_REFERENCE_DERIVATION = "PRIVATE_REFERENCE_DERIVATION"
    GRADER_RULE_DISCLOSURE = "GRADER_RULE_DISCLOSURE"
    HIDDEN_PASS_CONDITION_DISCLOSURE = "HIDDEN_PASS_CONDITION_DISCLOSURE"
    ORIGINAL_FINAL_OUTPUT_DERIVATION = "ORIGINAL_FINAL_OUTPUT_DERIVATION"
    EXECUTABILITY_FAILURE = "EXECUTABILITY_FAILURE"
    EVALUATOR_CONTRACT_MISMATCH = "EVALUATOR_CONTRACT_MISMATCH"
    TOOL_POLICY_MISMATCH = "TOOL_POLICY_MISMATCH"


class AttachmentSemanticFindingPolicyV2(FacadeModel):
    round: AttachmentSemanticReviewRoundV2
    severity: AttachmentSemanticReviewSeverityV2
    scope: AttachmentSemanticFindingScopeV2
    non_waivable: bool
    artifact_repair_allowed: bool


def _finding_policy(
    round_: AttachmentSemanticReviewRoundV2,
    severity: AttachmentSemanticReviewSeverityV2,
    scope: AttachmentSemanticFindingScopeV2,
    *,
    non_waivable: bool = False,
    repair: bool = False,
) -> AttachmentSemanticFindingPolicyV2:
    return AttachmentSemanticFindingPolicyV2(
        round=round_,
        severity=severity,
        scope=scope,
        non_waivable=non_waivable,
        artifact_repair_allowed=repair,
    )


_FINDING_POLICIES: dict[
    AttachmentSemanticReviewFindingCodeV2,
    AttachmentSemanticFindingPolicyV2,
] = {
    AttachmentSemanticReviewFindingCodeV2.REQUIRED_COVERAGE_MISSING: _finding_policy(
        AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY,
        AttachmentSemanticReviewSeverityV2.P1,
        AttachmentSemanticFindingScopeV2.ARTIFACT,
        repair=True,
    ),
    AttachmentSemanticReviewFindingCodeV2.DEPENDENCY_NOT_SOLVABLE: _finding_policy(
        AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY,
        AttachmentSemanticReviewSeverityV2.P1,
        AttachmentSemanticFindingScopeV2.ITEM,
    ),
    AttachmentSemanticReviewFindingCodeV2.PUBLIC_RUBRIC_UNREACHABLE: _finding_policy(
        AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY,
        AttachmentSemanticReviewSeverityV2.P1,
        AttachmentSemanticFindingScopeV2.ITEM,
    ),
    AttachmentSemanticReviewFindingCodeV2.SAFE_EVIDENCE_INSUFFICIENT: _finding_policy(
        AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY,
        AttachmentSemanticReviewSeverityV2.P1,
        AttachmentSemanticFindingScopeV2.ITEM,
    ),
    AttachmentSemanticReviewFindingCodeV2.CAPABILITY_CONTRACT_MISMATCH: _finding_policy(
        AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY,
        AttachmentSemanticReviewSeverityV2.P1,
        AttachmentSemanticFindingScopeV2.ITEM,
    ),
    AttachmentSemanticReviewFindingCodeV2.WORLD_FACT_INCONSISTENT: _finding_policy(
        AttachmentSemanticReviewRoundV2.REALISM_CONSISTENCY,
        AttachmentSemanticReviewSeverityV2.P1,
        AttachmentSemanticFindingScopeV2.ARTIFACT,
        repair=True,
    ),
    AttachmentSemanticReviewFindingCodeV2.SOURCE_EVIDENCE_INCONSISTENT: _finding_policy(
        AttachmentSemanticReviewRoundV2.REALISM_CONSISTENCY,
        AttachmentSemanticReviewSeverityV2.P1,
        AttachmentSemanticFindingScopeV2.ARTIFACT,
        repair=True,
    ),
    AttachmentSemanticReviewFindingCodeV2.ARTIFACT_CONTENT_UNREALISTIC: _finding_policy(
        AttachmentSemanticReviewRoundV2.REALISM_CONSISTENCY,
        AttachmentSemanticReviewSeverityV2.P1,
        AttachmentSemanticFindingScopeV2.ARTIFACT,
        repair=True,
    ),
    AttachmentSemanticReviewFindingCodeV2.CROSS_ARTIFACT_INCONSISTENT: _finding_policy(
        AttachmentSemanticReviewRoundV2.REALISM_CONSISTENCY,
        AttachmentSemanticReviewSeverityV2.P1,
        AttachmentSemanticFindingScopeV2.ARTIFACT,
        repair=True,
    ),
    AttachmentSemanticReviewFindingCodeV2.ARTIFACT_METADATA_INCONSISTENT: _finding_policy(
        AttachmentSemanticReviewRoundV2.REALISM_CONSISTENCY,
        AttachmentSemanticReviewSeverityV2.P2,
        AttachmentSemanticFindingScopeV2.ARTIFACT,
        repair=True,
    ),
    **{
        code: _finding_policy(
            AttachmentSemanticReviewRoundV2.LEAKAGE_EXECUTABILITY,
            AttachmentSemanticReviewSeverityV2.P0,
            AttachmentSemanticFindingScopeV2.ITEM,
            non_waivable=True,
        )
        for code in (
            AttachmentSemanticReviewFindingCodeV2.ANSWER_BEARING_CONTENT,
            AttachmentSemanticReviewFindingCodeV2.COMPLETED_DELIVERABLE_EXPOSED,
            AttachmentSemanticReviewFindingCodeV2.PRIVATE_REFERENCE_DERIVATION,
            AttachmentSemanticReviewFindingCodeV2.GRADER_RULE_DISCLOSURE,
            AttachmentSemanticReviewFindingCodeV2.HIDDEN_PASS_CONDITION_DISCLOSURE,
            AttachmentSemanticReviewFindingCodeV2.ORIGINAL_FINAL_OUTPUT_DERIVATION,
        )
    },
    AttachmentSemanticReviewFindingCodeV2.EXECUTABILITY_FAILURE: _finding_policy(
        AttachmentSemanticReviewRoundV2.LEAKAGE_EXECUTABILITY,
        AttachmentSemanticReviewSeverityV2.P1,
        AttachmentSemanticFindingScopeV2.ARTIFACT,
        repair=True,
    ),
    AttachmentSemanticReviewFindingCodeV2.EVALUATOR_CONTRACT_MISMATCH: _finding_policy(
        AttachmentSemanticReviewRoundV2.LEAKAGE_EXECUTABILITY,
        AttachmentSemanticReviewSeverityV2.P1,
        AttachmentSemanticFindingScopeV2.ITEM,
    ),
    AttachmentSemanticReviewFindingCodeV2.TOOL_POLICY_MISMATCH: _finding_policy(
        AttachmentSemanticReviewRoundV2.LEAKAGE_EXECUTABILITY,
        AttachmentSemanticReviewSeverityV2.P1,
        AttachmentSemanticFindingScopeV2.ITEM,
    ),
}


def attachment_semantic_finding_policy(
    code: AttachmentSemanticReviewFindingCodeV2,
) -> AttachmentSemanticFindingPolicyV2:
    return _FINDING_POLICIES[code]


class AttachmentSemanticReviewOutcomeV2(StrEnum):
    ACCEPTED = "ACCEPTED"
    REQUIRES_REPAIR = "REQUIRES_REPAIR"
    REJECTED = "REJECTED"
    ABSTAINED = "ABSTAINED"
    BLOCKED = "BLOCKED"


class AttachmentSemanticReviewFailureCodeV2(StrEnum):
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    CONTEXT_MISSING = "CONTEXT_MISSING"
    CONTEXT_MISMATCH = "CONTEXT_MISMATCH"
    BACKEND_FAILED = "BACKEND_FAILED"
    BACKEND_OUTPUT_INVALID = "BACKEND_OUTPUT_INVALID"
    OUTPUT_HASH_MISMATCH = "OUTPUT_HASH_MISMATCH"


class AttachmentSemanticResolutionDispositionV2(StrEnum):
    CONFIRMED_RESOLVED = "CONFIRMED_RESOLVED"
    PERSISTS = "PERSISTS"


class AttachmentRepairSourceV2(StrEnum):
    DETERMINISTIC = "DETERMINISTIC"
    COVERAGE_SOLVABILITY = "COVERAGE_SOLVABILITY"
    REALISM_CONSISTENCY = "REALISM_CONSISTENCY"


class AttachmentRepairOutcomeV2(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_POLICY = "BLOCKED_POLICY"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    TERMINAL_FAILURE = "TERMINAL_FAILURE"


class AttachmentRepairFailureCodeV2(StrEnum):
    REPAIR_BACKEND_UNAVAILABLE = "REPAIR_BACKEND_UNAVAILABLE"
    REPAIR_MATERIAL_MISSING = "REPAIR_MATERIAL_MISSING"
    REPAIR_MATERIAL_MISMATCH = "REPAIR_MATERIAL_MISMATCH"
    REPAIR_NOT_AUTHORIZED = "REPAIR_NOT_AUTHORIZED"
    REPAIR_BACKEND_FAILED = "REPAIR_BACKEND_FAILED"
    OUTPUT_UNCHANGED = "OUTPUT_UNCHANGED"
    OUTPUT_INVALID = "OUTPUT_INVALID"
    VALIDATION_REGISTRATION_UNAVAILABLE = "VALIDATION_REGISTRATION_UNAVAILABLE"
    VALIDATION_REGISTRATION_FAILED = "VALIDATION_REGISTRATION_FAILED"


class AttachmentSemanticReviewRequestV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-semantic-review-request/v2"] = (
        "env-mock-agent/attachment-semantic-review-request/v2"
    )
    semantic_review_request_id: Identifier
    round: AttachmentSemanticReviewRoundV2
    reviewer_role: AttachmentSemanticReviewerRoleV2
    stage_run_ref: FacadeObjectRef
    candidate_revision_ref: FacadeObjectRef
    deterministic_validation_result_ref: FacadeObjectRef
    context_view_ref: FacadeObjectRef
    current_artifact_version_refs: tuple[FacadeObjectRef, ...]
    current_output_refs: tuple[FacadeObjectRef, ...]
    prior_round_result_ref: FacadeObjectRef | None = None
    prior_finding_refs: tuple[FacadeObjectRef, ...] = ()
    required_resolution_finding_refs: tuple[FacadeObjectRef, ...] = ()
    prior_resolution_refs: tuple[FacadeObjectRef, ...] = ()
    prior_repair_plan_refs: tuple[FacadeObjectRef, ...] = ()
    prior_repair_result_refs: tuple[FacadeObjectRef, ...] = ()
    model_profile_ref: FacadeObjectRef
    prompt_version: str
    policy_version: Literal["semantic-review/r5-09-v1"] = ATTACHMENT_SEMANTIC_REVIEW_POLICY_VERSION
    idempotency_key: Identifier
    semantic_review_request_sha256: Sha256

    @model_validator(mode="after")
    def validate_request(self) -> AttachmentSemanticReviewRequestV2:
        if self.reviewer_role is not ROUND_ROLES[self.round]:
            raise ValueError("semantic review round and role do not match")
        for ref, object_type, field_name in (
            (self.stage_run_ref, "stage-run", "stage_run_ref"),
            (
                self.candidate_revision_ref,
                "attachment-candidate-revision",
                "candidate_revision_ref",
            ),
            (
                self.deterministic_validation_result_ref,
                "revision-deterministic-validation",
                "deterministic_validation_result_ref",
            ),
            (
                self.context_view_ref,
                "semantic-review-context-view",
                "context_view_ref",
            ),
            (self.model_profile_ref, "model-profile", "model_profile_ref"),
        ):
            _require_ref(ref, object_type, field_name)
        for ref in self.current_artifact_version_refs:
            _require_ref(
                ref,
                "candidate-artifact-version",
                "current_artifact_version_refs",
            )
        for ref in self.current_output_refs:
            _require_ref(ref, "attachment-output", "current_output_refs")
        if self.round is AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY:
            if self.prior_round_result_ref is not None:
                raise ValueError("first round cannot carry predecessor")
        elif self.prior_round_result_ref is None:
            raise ValueError("later semantic round requires predecessor")
        else:
            _require_ref(
                self.prior_round_result_ref,
                "semantic-review-round-result",
                "prior_round_result_ref",
            )
        for label, refs in (
            ("current artifact version refs", self.current_artifact_version_refs),
            ("current output refs", self.current_output_refs),
            ("prior finding refs", self.prior_finding_refs),
            (
                "required resolution finding refs",
                self.required_resolution_finding_refs,
            ),
            ("prior resolution refs", self.prior_resolution_refs),
            ("prior repair plan refs", self.prior_repair_plan_refs),
            ("prior repair result refs", self.prior_repair_result_refs),
        ):
            _require_sorted_unique_refs(label, refs)
        for ref in self.prior_repair_plan_refs:
            _require_ref(ref, "targeted-repair-plan", "prior_repair_plan_refs")
        for ref in self.prior_repair_result_refs:
            _require_ref(
                ref,
                "attachment-repair-result",
                "prior_repair_result_refs",
            )
        for ref in self.prior_finding_refs:
            _require_ref(
                ref,
                "attachment-semantic-review-finding",
                "prior_finding_refs",
            )
        for ref in self.required_resolution_finding_refs:
            _require_ref(
                ref,
                "attachment-semantic-review-finding",
                "required_resolution_finding_refs",
            )
        if not set(self.required_resolution_finding_refs) <= set(self.prior_finding_refs):
            raise ValueError("required resolution findings must be prior findings")
        for ref in self.prior_resolution_refs:
            _require_ref(
                ref,
                "attachment-semantic-finding-resolution",
                "prior_resolution_refs",
            )
        if bool(self.prior_repair_plan_refs) != bool(self.prior_repair_result_refs):
            raise ValueError("prior repair plan and result refs must be present together")
        if self.round is AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY and (
            self.prior_finding_refs
            or self.required_resolution_finding_refs
            or self.prior_resolution_refs
            or self.prior_repair_plan_refs
            or self.prior_repair_result_refs
        ):
            raise ValueError("first round cannot carry prior review or repair facts")
        _validate_final_identity(
            object_id=self.semantic_review_request_id,
            object_sha256=self.semantic_review_request_sha256,
            expected_prefix="attachment-semantic-review-request",
            observed=attachment_semantic_review_request_carried_sha256(self),
        )
        return self


class AttachmentSemanticReviewFindingV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-semantic-review-finding/v2"] = (
        "env-mock-agent/attachment-semantic-review-finding/v2"
    )
    finding_id: Identifier
    semantic_review_request_ref: FacadeObjectRef
    round: AttachmentSemanticReviewRoundV2
    scope: AttachmentSemanticFindingScopeV2
    code: AttachmentSemanticReviewFindingCodeV2
    candidate_revision_ref: FacadeObjectRef
    subject_refs: tuple[FacadeObjectRef, ...]
    artifact_ids: tuple[Identifier, ...]
    evidence_ref_ids: tuple[Identifier, ...]
    predecessor_finding_ref: FacadeObjectRef | None = None
    severity: AttachmentSemanticReviewSeverityV2
    non_waivable: bool
    artifact_repair_allowed: bool
    finding_sha256: Sha256

    @model_validator(mode="after")
    def validate_finding(self) -> AttachmentSemanticReviewFindingV2:
        policy = attachment_semantic_finding_policy(self.code)
        if (
            self.round is not policy.round
            or self.scope is not policy.scope
            or self.severity is not policy.severity
            or self.non_waivable is not policy.non_waivable
            or self.artifact_repair_allowed is not policy.artifact_repair_allowed
        ):
            raise ValueError("semantic finding policy does not match code")
        _require_ref(
            self.semantic_review_request_ref,
            "attachment-semantic-review-request",
            "semantic_review_request_ref",
        )
        _require_ref(
            self.candidate_revision_ref,
            "attachment-candidate-revision",
            "candidate_revision_ref",
        )
        _require_sorted_unique_refs("finding subject refs", self.subject_refs)
        _require_sorted_unique("finding artifact IDs", self.artifact_ids)
        _require_sorted_unique("finding evidence IDs", self.evidence_ref_ids)
        if self.scope is AttachmentSemanticFindingScopeV2.ARTIFACT:
            if not self.artifact_ids:
                raise ValueError("artifact semantic finding requires artifact IDs")
        elif self.artifact_ids:
            raise ValueError("item semantic finding cannot carry artifact IDs")
        _validate_final_identity(
            object_id=self.finding_id,
            object_sha256=self.finding_sha256,
            expected_prefix="attachment-semantic-review-finding",
            observed=attachment_semantic_review_finding_carried_sha256(self),
        )
        return self


class AttachmentSemanticFindingResolutionV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-semantic-finding-resolution/v2"] = (
        "env-mock-agent/attachment-semantic-finding-resolution/v2"
    )
    resolution_id: Identifier
    semantic_review_request_ref: FacadeObjectRef
    prior_finding_ref: FacadeObjectRef
    old_subject_refs: tuple[FacadeObjectRef, ...]
    current_subject_refs: tuple[FacadeObjectRef, ...]
    repair_plan_ref: FacadeObjectRef
    repair_result_refs: tuple[FacadeObjectRef, ...]
    deterministic_revalidation_ref: FacadeObjectRef
    disposition: AttachmentSemanticResolutionDispositionV2
    successor_finding_ref: FacadeObjectRef | None = None
    evidence_ref_ids: tuple[Identifier, ...]
    resolution_sha256: Sha256

    @model_validator(mode="after")
    def validate_resolution(self) -> AttachmentSemanticFindingResolutionV2:
        _require_ref(
            self.semantic_review_request_ref,
            "attachment-semantic-review-request",
            "semantic_review_request_ref",
        )
        _require_ref(
            self.prior_finding_ref,
            "attachment-semantic-review-finding",
            "prior_finding_ref",
        )
        _require_ref(
            self.repair_plan_ref,
            "targeted-repair-plan",
            "repair_plan_ref",
        )
        _require_ref(
            self.deterministic_revalidation_ref,
            "revision-deterministic-validation",
            "deterministic_revalidation_ref",
        )
        _require_sorted_unique_refs(
            "resolution old subject refs",
            self.old_subject_refs,
        )
        _require_sorted_unique_refs(
            "resolution current subject refs",
            self.current_subject_refs,
        )
        _require_sorted_unique_refs(
            "resolution repair result refs",
            self.repair_result_refs,
        )
        _require_sorted_unique(
            "resolution evidence IDs",
            self.evidence_ref_ids,
        )
        if (
            not self.old_subject_refs
            or not self.current_subject_refs
            or self.old_subject_refs == self.current_subject_refs
            or not self.repair_result_refs
        ):
            raise ValueError("semantic resolution requires changed subjects and repair results")
        if self.disposition is AttachmentSemanticResolutionDispositionV2.CONFIRMED_RESOLVED:
            if self.successor_finding_ref is not None:
                raise ValueError("resolved finding cannot carry successor")
        elif self.successor_finding_ref is None:
            raise ValueError("persistent finding requires successor")
        else:
            _require_ref(
                self.successor_finding_ref,
                "attachment-semantic-review-finding",
                "successor_finding_ref",
            )
        _validate_final_identity(
            object_id=self.resolution_id,
            object_sha256=self.resolution_sha256,
            expected_prefix="attachment-semantic-finding-resolution",
            observed=attachment_semantic_finding_resolution_carried_sha256(self),
        )
        return self


class SemanticCleanContextAttestationV2(FacadeModel):
    schema_version: Literal["env-mock-agent/semantic-clean-context-attestation/v2"] = (
        "env-mock-agent/semantic-clean-context-attestation/v2"
    )
    attestation_id: Identifier
    semantic_review_request_ref: FacadeObjectRef
    stage_run_ref: FacadeObjectRef
    round: AttachmentSemanticReviewRoundV2
    reviewer_role: AttachmentSemanticReviewerRoleV2
    context_view_ref: FacadeObjectRef
    included_ref_inventory: tuple[FacadeObjectRef, ...]
    denied_data_families: tuple[Identifier, ...]
    fresh_context: Literal[True] = True
    runtime_resume_used: Literal[False] = False
    build_transcript_included: Literal[False] = False
    hidden_reasoning_included: Literal[False] = False
    raw_private_reference_included: Literal[False] = False
    policy_version: Literal["semantic-review/r5-09-v1"] = ATTACHMENT_SEMANTIC_REVIEW_POLICY_VERSION
    attestation_sha256: Sha256

    @classmethod
    def create(
        cls,
        *,
        semantic_review_request_ref: FacadeObjectRef,
        stage_run_ref: FacadeObjectRef,
        round: AttachmentSemanticReviewRoundV2,
        reviewer_role: AttachmentSemanticReviewerRoleV2,
        context_view_ref: FacadeObjectRef,
        included_ref_inventory: tuple[FacadeObjectRef, ...],
        denied_data_families: tuple[Identifier, ...] = (
            "BUILD_TRANSCRIPT",
            "HIDDEN_REASONING",
            "RAW_PRIVATE_REFERENCE",
            "RAW_TRACE",
        ),
    ) -> SemanticCleanContextAttestationV2:
        value = cls(
            attestation_id="semantic-clean-context-attestation://pending",
            semantic_review_request_ref=semantic_review_request_ref,
            stage_run_ref=stage_run_ref,
            round=round,
            reviewer_role=reviewer_role,
            context_view_ref=context_view_ref,
            included_ref_inventory=_sorted_refs(included_ref_inventory),
            denied_data_families=tuple(sorted(denied_data_families)),
            attestation_sha256="0" * 64,
        )
        digest = semantic_clean_context_attestation_carried_sha256(value)
        return value.model_copy(
            update={
                "attestation_id": (f"semantic-clean-context-attestation://sha256/{digest}"),
                "attestation_sha256": digest,
            }
        )


class AttachmentSemanticReviewResultV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-semantic-review-result/v2"] = (
        "env-mock-agent/attachment-semantic-review-result/v2"
    )
    semantic_review_result_id: Identifier
    semantic_review_request_ref: FacadeObjectRef
    candidate_revision_ref: FacadeObjectRef
    deterministic_validation_result_ref: FacadeObjectRef
    round: AttachmentSemanticReviewRoundV2
    reviewer_role: AttachmentSemanticReviewerRoleV2
    outcome: AttachmentSemanticReviewOutcomeV2
    clean_context_attestation: SemanticCleanContextAttestationV2
    findings: tuple[AttachmentSemanticReviewFindingV2, ...] = ()
    resolutions: tuple[AttachmentSemanticFindingResolutionV2, ...] = ()
    failure_code: AttachmentSemanticReviewFailureCodeV2 | None = None
    policy_version: Literal["semantic-review/r5-09-v1"] = ATTACHMENT_SEMANTIC_REVIEW_POLICY_VERSION
    semantic_review_result_sha256: Sha256

    @model_validator(mode="after")
    def validate_result(self) -> AttachmentSemanticReviewResultV2:
        if self.reviewer_role is not ROUND_ROLES[self.round]:
            raise ValueError("semantic review result round and role do not match")
        if any(
            item.round is not self.round
            or item.candidate_revision_ref != self.candidate_revision_ref
            or item.semantic_review_request_ref != self.semantic_review_request_ref
            for item in self.findings
        ):
            raise ValueError("semantic review findings do not match exact result")
        if any(
            item.semantic_review_request_ref != self.semantic_review_request_ref for item in self.resolutions
        ):
            raise ValueError("semantic resolutions do not match exact result")
        blockers = tuple(
            item
            for item in self.findings
            if item.severity
            in {
                AttachmentSemanticReviewSeverityV2.P0,
                AttachmentSemanticReviewSeverityV2.P1,
            }
            or item.non_waivable
        )
        if self.outcome is AttachmentSemanticReviewOutcomeV2.ACCEPTED:
            if blockers or self.failure_code is not None:
                raise ValueError("ACCEPTED semantic review cannot contain blockers")
        elif self.outcome is AttachmentSemanticReviewOutcomeV2.REQUIRES_REPAIR:
            if not blockers or any(item.non_waivable for item in blockers) or self.failure_code is not None:
                raise ValueError("REQUIRES_REPAIR semantic review is inconsistent")
        elif self.outcome is AttachmentSemanticReviewOutcomeV2.REJECTED:
            if not any(item.non_waivable for item in blockers) or self.failure_code is not None:
                raise ValueError("REJECTED semantic review requires non-waivable finding")
        elif self.failure_code is None or self.findings or self.resolutions:
            raise ValueError("ABSTAINED or BLOCKED semantic review requires only a failure code")
        _validate_final_identity(
            object_id=self.semantic_review_result_id,
            object_sha256=self.semantic_review_result_sha256,
            expected_prefix="attachment-semantic-review-result",
            observed=attachment_semantic_review_result_carried_sha256(self),
        )
        return self


class AttachmentRepairRequestV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-repair-request/v2"] = (
        "env-mock-agent/attachment-repair-request/v2"
    )
    repair_request_id: Identifier
    repair_plan_ref: FacadeObjectRef
    source_round: AttachmentRepairSourceV2
    candidate_revision_ref: FacadeObjectRef
    artifact_id: Identifier
    artifact_version_ref: FacadeObjectRef
    build_spec_ref: FacadeObjectRef
    execution_result_ref: FacadeObjectRef
    output_ref: FacadeObjectRef
    output_sha256: Sha256
    targeted_finding_refs: tuple[FacadeObjectRef, ...]
    targeted_finding_codes: tuple[Identifier, ...]
    attempt: int
    prior_repair_result_ref: FacadeObjectRef | None = None
    policy_version: Literal["targeted-repair/r5-09-v1"] = ATTACHMENT_REPAIR_POLICY_VERSION
    idempotency_key: Identifier
    repair_request_sha256: Sha256

    @model_validator(mode="after")
    def validate_request(self) -> AttachmentRepairRequestV2:
        if self.output_ref.object_sha256 != self.output_sha256:
            raise ValueError("repair output ref/hash mismatch")
        if not self.targeted_finding_refs or not self.targeted_finding_codes:
            raise ValueError("repair request requires targeted findings")
        if len(self.targeted_finding_refs) != len(self.targeted_finding_codes):
            raise ValueError("repair finding refs/codes must align")
        if self.source_round is not AttachmentRepairSourceV2.DETERMINISTIC:
            try:
                semantic_codes = tuple(
                    AttachmentSemanticReviewFindingCodeV2(code) for code in self.targeted_finding_codes
                )
            except ValueError as exc:
                raise ValueError("semantic repair request contains unknown finding code") from exc
            if any(
                not attachment_semantic_finding_policy(code).artifact_repair_allowed
                for code in semantic_codes
            ):
                raise ValueError("repair request contains non-repairable finding")
        if self.attempt == 1 and self.prior_repair_result_ref is not None:
            raise ValueError("first repair attempt cannot carry predecessor")
        if self.attempt > 1 and self.prior_repair_result_ref is None:
            raise ValueError("later repair attempt requires predecessor")
        _validate_final_identity(
            object_id=self.repair_request_id,
            object_sha256=self.repair_request_sha256,
            expected_prefix="attachment-repair-request",
            observed=attachment_repair_request_carried_sha256(self),
        )
        return self


class AttachmentRepairResultV2(FacadeModel):
    schema_version: Literal["env-mock-agent/attachment-repair-result/v2"] = (
        "env-mock-agent/attachment-repair-result/v2"
    )
    repair_result_id: Identifier
    repair_request_ref: FacadeObjectRef
    candidate_revision_ref: FacadeObjectRef
    artifact_id: Identifier
    source_artifact_version_ref: FacadeObjectRef
    source_output_ref: FacadeObjectRef
    targeted_finding_refs: tuple[FacadeObjectRef, ...]
    attempt: int
    outcome: AttachmentRepairOutcomeV2
    worker_version: str | None = None
    output_ref: FacadeObjectRef | None = None
    output_sha256: Sha256 | None = None
    retryable: bool
    failure_code: AttachmentRepairFailureCodeV2 | None = None
    policy_version: Literal["targeted-repair/r5-09-v1"] = ATTACHMENT_REPAIR_POLICY_VERSION
    repair_result_sha256: Sha256

    @model_validator(mode="after")
    def validate_result(self) -> AttachmentRepairResultV2:
        if self.outcome is AttachmentRepairOutcomeV2.SUCCEEDED:
            if (
                self.output_ref is None
                or self.output_sha256 is None
                or self.output_ref.object_sha256 != self.output_sha256
                or self.output_ref.object_sha256 == self.source_output_ref.object_sha256
                or self.worker_version is None
                or self.failure_code is not None
                or self.retryable
            ):
                raise ValueError("successful repair result is incomplete or unchanged")
        else:
            if (
                self.output_ref is not None
                or self.output_sha256 is not None
                or self.failure_code is None
                or self.worker_version is not None
            ):
                raise ValueError("failed repair result has invalid output fields")
            if self.retryable is not (self.outcome is AttachmentRepairOutcomeV2.RETRYABLE_FAILURE):
                raise ValueError("repair retryable flag does not match outcome")
        _validate_final_identity(
            object_id=self.repair_result_id,
            object_sha256=self.repair_result_sha256,
            expected_prefix="attachment-repair-result",
            observed=attachment_repair_result_carried_sha256(self),
        )
        return self


class AttachmentSemanticReviewFacade(Protocol):
    async def review(
        self,
        request: AttachmentSemanticReviewRequestV2,
    ) -> AttachmentSemanticReviewResultV2: ...

    async def repair(
        self,
        request: AttachmentRepairRequestV2,
    ) -> AttachmentRepairResultV2: ...


def attachment_semantic_review_request_carried_sha256(
    value: AttachmentSemanticReviewRequestV2,
) -> str:
    return _carried(value, "semantic_review_request_id", "semantic_review_request_sha256")


def attachment_semantic_review_request_ref(
    value: AttachmentSemanticReviewRequestV2,
) -> FacadeObjectRef:
    return _ref(
        value.semantic_review_request_id,
        "attachment-semantic-review-request",
        value.semantic_review_request_sha256,
    )


def attachment_semantic_review_finding_carried_sha256(
    value: AttachmentSemanticReviewFindingV2,
) -> str:
    return _carried(value, "finding_id", "finding_sha256")


def attachment_semantic_review_finding_ref(
    value: AttachmentSemanticReviewFindingV2,
) -> FacadeObjectRef:
    return _ref(value.finding_id, "attachment-semantic-review-finding", value.finding_sha256)


def attachment_semantic_finding_resolution_carried_sha256(
    value: AttachmentSemanticFindingResolutionV2,
) -> str:
    return _carried(value, "resolution_id", "resolution_sha256")


def attachment_semantic_finding_resolution_ref(
    value: AttachmentSemanticFindingResolutionV2,
) -> FacadeObjectRef:
    return _ref(
        value.resolution_id,
        "attachment-semantic-finding-resolution",
        value.resolution_sha256,
    )


def semantic_clean_context_attestation_carried_sha256(
    value: SemanticCleanContextAttestationV2,
) -> str:
    return _carried(value, "attestation_id", "attestation_sha256")


def semantic_clean_context_attestation_ref(
    value: SemanticCleanContextAttestationV2,
) -> FacadeObjectRef:
    return _ref(value.attestation_id, "semantic-clean-context-attestation", value.attestation_sha256)


def attachment_semantic_review_result_carried_sha256(
    value: AttachmentSemanticReviewResultV2,
) -> str:
    return _carried(value, "semantic_review_result_id", "semantic_review_result_sha256")


def attachment_semantic_review_result_ref(
    value: AttachmentSemanticReviewResultV2,
) -> FacadeObjectRef:
    return _ref(
        value.semantic_review_result_id,
        "attachment-semantic-review-result",
        value.semantic_review_result_sha256,
    )


def attachment_repair_request_carried_sha256(
    value: AttachmentRepairRequestV2,
) -> str:
    return _carried(value, "repair_request_id", "repair_request_sha256")


def attachment_repair_request_ref(
    value: AttachmentRepairRequestV2,
) -> FacadeObjectRef:
    return _ref(value.repair_request_id, "attachment-repair-request", value.repair_request_sha256)


def attachment_repair_result_carried_sha256(
    value: AttachmentRepairResultV2,
) -> str:
    return _carried(value, "repair_result_id", "repair_result_sha256")


def attachment_repair_result_ref(
    value: AttachmentRepairResultV2,
) -> FacadeObjectRef:
    return _ref(value.repair_result_id, "attachment-repair-result", value.repair_result_sha256)


def validate_attachment_semantic_review_request_identity(
    value: AttachmentSemanticReviewRequestV2,
) -> None:
    _validate_identity(
        value.semantic_review_request_id,
        value.semantic_review_request_sha256,
        "attachment-semantic-review-request",
        attachment_semantic_review_request_carried_sha256(value),
    )


def validate_attachment_semantic_review_result_identity(
    value: AttachmentSemanticReviewResultV2,
) -> None:
    _validate_identity(
        value.semantic_review_result_id,
        value.semantic_review_result_sha256,
        "attachment-semantic-review-result",
        attachment_semantic_review_result_carried_sha256(value),
    )


def validate_attachment_repair_request_identity(value: AttachmentRepairRequestV2) -> None:
    _validate_identity(
        value.repair_request_id,
        value.repair_request_sha256,
        "attachment-repair-request",
        attachment_repair_request_carried_sha256(value),
    )


def _carried(value: FacadeModel, *excluded: str) -> str:
    return _payload_sha256(value.model_dump(mode="json", exclude=set(excluded), exclude_none=False))


def _ref(object_id: str, object_type: str, digest: str) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version="v2",
        object_sha256=digest,
    )


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _validate_final_identity(
    *,
    object_id: str,
    object_sha256: str,
    expected_prefix: str,
    observed: str,
) -> None:
    if object_id.endswith("://pending") or object_sha256 == "0" * 64:
        return
    _validate_identity(object_id, object_sha256, expected_prefix, observed)


def _validate_identity(
    object_id: str,
    object_sha256: str,
    expected_prefix: str,
    observed: str,
) -> None:
    if object_sha256 != observed or object_id != f"{expected_prefix}://sha256/{observed}":
        raise ValueError(f"{expected_prefix} identity is stale")


def _require_ref(
    ref: FacadeObjectRef,
    object_type: str,
    field_name: str,
) -> None:
    if ref.object_type != object_type:
        raise ValueError(f"{field_name} must reference {object_type}")


def _ref_key(ref: FacadeObjectRef) -> tuple[str, str, str, str]:
    return (
        ref.object_type,
        ref.object_id,
        ref.object_version,
        ref.object_sha256,
    )


def _sorted_refs(
    refs: tuple[FacadeObjectRef, ...],
) -> tuple[FacadeObjectRef, ...]:
    unique = {_ref_key(ref): ref for ref in refs}
    return tuple(unique[key] for key in sorted(unique))


def _require_sorted_unique_refs(
    label: str,
    refs: tuple[FacadeObjectRef, ...],
) -> None:
    if refs != _sorted_refs(refs) or len(refs) != len(set(refs)):
        raise ValueError(f"{label} must be sorted and unique")


def _require_sorted_unique(label: str, values: tuple[object, ...]) -> None:
    if len(values) != len(set(values)) or values != tuple(sorted(values, key=repr)):
        raise ValueError(f"{label} must be sorted and unique")
