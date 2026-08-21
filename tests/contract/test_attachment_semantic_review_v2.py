from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from env_mock_agent.facade import FacadeObjectRef
from env_mock_agent.facade.semantic_review_v2 import (
    ATTACHMENT_SEMANTIC_REVIEW_POLICY_VERSION,
    AttachmentSemanticFindingResolutionV2,
    AttachmentSemanticFindingScopeV2,
    AttachmentSemanticResolutionDispositionV2,
    AttachmentSemanticReviewerRoleV2,
    AttachmentSemanticReviewFindingCodeV2,
    AttachmentSemanticReviewFindingV2,
    AttachmentSemanticReviewOutcomeV2,
    AttachmentSemanticReviewRequestV2,
    AttachmentSemanticReviewResultV2,
    AttachmentSemanticReviewRoundV2,
    AttachmentSemanticReviewSeverityV2,
    SemanticCleanContextAttestationV2,
    attachment_semantic_finding_policy,
    attachment_semantic_finding_resolution_carried_sha256,
    attachment_semantic_finding_resolution_ref,
    attachment_semantic_review_finding_carried_sha256,
    attachment_semantic_review_request_carried_sha256,
    attachment_semantic_review_request_ref,
    attachment_semantic_review_result_carried_sha256,
    validate_attachment_semantic_review_request_identity,
    validate_attachment_semantic_review_result_identity,
)

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str,
    *,
    digest: str = HASH,
    version: str = "v2",
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _request(
    round_: AttachmentSemanticReviewRoundV2 = (AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY),
) -> AttachmentSemanticReviewRequestV2:
    role = {
        AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY: (
            AttachmentSemanticReviewerRoleV2.COVERAGE_SOLVABILITY_REVIEWER
        ),
        AttachmentSemanticReviewRoundV2.REALISM_CONSISTENCY: (
            AttachmentSemanticReviewerRoleV2.REALISM_CONSISTENCY_REVIEWER
        ),
        AttachmentSemanticReviewRoundV2.LEAKAGE_EXECUTABILITY: (
            AttachmentSemanticReviewerRoleV2.LEAKAGE_EXECUTABILITY_REVIEWER
        ),
    }[round_]
    prior_ref = (
        None
        if round_ is AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY
        else _ref("semantic-review-round-result", "prior")
    )
    request = AttachmentSemanticReviewRequestV2(
        semantic_review_request_id="attachment-semantic-review-request://pending",
        round=round_,
        reviewer_role=role,
        stage_run_ref=_ref("stage-run", round_.value.lower(), version="identity/v1"),
        candidate_revision_ref=_ref("attachment-candidate-revision", "current"),
        deterministic_validation_result_ref=_ref(
            "revision-deterministic-validation",
            "current",
        ),
        context_view_ref=_ref("semantic-review-context-view", round_.value.lower()),
        current_artifact_version_refs=(_ref("candidate-artifact-version", "input"),),
        current_output_refs=(_ref("attachment-output", "input"),),
        prior_round_result_ref=prior_ref,
        prior_finding_refs=(),
        prior_resolution_refs=(),
        model_profile_ref=_ref("model-profile", round_.value.lower(), version="v1"),
        prompt_version=f"semantic-review/{round_.value.lower()}/v1",
        policy_version=ATTACHMENT_SEMANTIC_REVIEW_POLICY_VERSION,
        idempotency_key=f"semantic-review-idempotency://{round_.value.lower()}",
        semantic_review_request_sha256=HASH,
    )
    digest = attachment_semantic_review_request_carried_sha256(request)
    return request.model_copy(
        update={
            "semantic_review_request_id": (f"attachment-semantic-review-request://sha256/{digest}"),
            "semantic_review_request_sha256": digest,
        }
    )


def _finding(
    request: AttachmentSemanticReviewRequestV2,
    code: AttachmentSemanticReviewFindingCodeV2 = (
        AttachmentSemanticReviewFindingCodeV2.REQUIRED_COVERAGE_MISSING
    ),
) -> AttachmentSemanticReviewFindingV2:
    policy = attachment_semantic_finding_policy(code)
    artifact_scope = policy.scope is AttachmentSemanticFindingScopeV2.ARTIFACT
    subjects = (
        request.candidate_revision_ref,
        *(request.current_artifact_version_refs if artifact_scope else ()),
        *(request.current_output_refs if artifact_scope else ()),
    )
    finding = AttachmentSemanticReviewFindingV2(
        finding_id="attachment-semantic-review-finding://pending",
        semantic_review_request_ref=attachment_semantic_review_request_ref(request),
        round=request.round,
        scope=policy.scope,
        code=code,
        candidate_revision_ref=request.candidate_revision_ref,
        subject_refs=tuple(
            sorted(
                subjects,
                key=lambda ref: (
                    ref.object_type,
                    ref.object_id,
                    ref.object_version,
                    ref.object_sha256,
                ),
            )
        ),
        artifact_ids=("artifact://input",) if artifact_scope else (),
        evidence_ref_ids=("semantic-evidence://coverage/input",),
        predecessor_finding_ref=None,
        severity=policy.severity,
        non_waivable=policy.non_waivable,
        artifact_repair_allowed=policy.artifact_repair_allowed,
        finding_sha256=HASH,
    )
    digest = attachment_semantic_review_finding_carried_sha256(finding)
    return finding.model_copy(
        update={
            "finding_id": f"attachment-semantic-review-finding://sha256/{digest}",
            "finding_sha256": digest,
        }
    )


def _attestation(
    request: AttachmentSemanticReviewRequestV2,
) -> SemanticCleanContextAttestationV2:
    included = tuple(
        sorted(
            (
                request.candidate_revision_ref,
                request.deterministic_validation_result_ref,
                request.context_view_ref,
                *request.current_artifact_version_refs,
                *request.current_output_refs,
            ),
            key=lambda ref: (
                ref.object_type,
                ref.object_id,
                ref.object_version,
                ref.object_sha256,
            ),
        )
    )
    return SemanticCleanContextAttestationV2.create(
        semantic_review_request_ref=attachment_semantic_review_request_ref(request),
        stage_run_ref=request.stage_run_ref,
        round=request.round,
        reviewer_role=request.reviewer_role,
        context_view_ref=request.context_view_ref,
        included_ref_inventory=included,
    )


def _result(
    request: AttachmentSemanticReviewRequestV2,
    *,
    findings: tuple[AttachmentSemanticReviewFindingV2, ...] = (),
) -> AttachmentSemanticReviewResultV2:
    result = AttachmentSemanticReviewResultV2(
        semantic_review_result_id="attachment-semantic-review-result://pending",
        semantic_review_request_ref=attachment_semantic_review_request_ref(request),
        candidate_revision_ref=request.candidate_revision_ref,
        deterministic_validation_result_ref=request.deterministic_validation_result_ref,
        round=request.round,
        reviewer_role=request.reviewer_role,
        outcome=(
            AttachmentSemanticReviewOutcomeV2.REQUIRES_REPAIR
            if findings
            else AttachmentSemanticReviewOutcomeV2.ACCEPTED
        ),
        clean_context_attestation=_attestation(request),
        findings=findings,
        resolutions=(),
        failure_code=None,
        policy_version=ATTACHMENT_SEMANTIC_REVIEW_POLICY_VERSION,
        semantic_review_result_sha256=HASH,
    )
    digest = attachment_semantic_review_result_carried_sha256(result)
    return result.model_copy(
        update={
            "semantic_review_result_id": (f"attachment-semantic-review-result://sha256/{digest}"),
            "semantic_review_result_sha256": digest,
        }
    )


def test_review_request_is_strict_role_bound_and_content_free() -> None:
    request = _request()

    validate_attachment_semantic_review_request_identity(request)
    assert request.reviewer_role is (AttachmentSemanticReviewerRoleV2.COVERAGE_SOLVABILITY_REVIEWER)
    serialized = json.dumps(request.model_dump(mode="json"), sort_keys=True)
    for denied in (
        "artifact_path",
        "artifact_content",
        "private_reference",
        "grader_rule",
        "runtime_transcript",
        "hidden_reasoning",
    ):
        assert denied not in serialized

    values = request.model_dump(mode="python")
    values["reviewer_role"] = AttachmentSemanticReviewerRoleV2.LEAKAGE_EXECUTABILITY_REVIEWER
    with pytest.raises(ValidationError, match=r"round.*role"):
        AttachmentSemanticReviewRequestV2.model_validate(values)


def test_round_predecessor_matrix_is_closed() -> None:
    first = _request()
    values = first.model_dump(mode="python")
    values["prior_round_result_ref"] = _ref("semantic-review-round-result", "illegal")
    with pytest.raises(ValidationError, match="first round"):
        AttachmentSemanticReviewRequestV2.model_validate(values)

    second = _request(AttachmentSemanticReviewRoundV2.REALISM_CONSISTENCY)
    values = second.model_dump(mode="python")
    values["prior_round_result_ref"] = None
    with pytest.raises(ValidationError, match="predecessor"):
        AttachmentSemanticReviewRequestV2.model_validate(values)


def test_finding_policy_cannot_be_downgraded() -> None:
    request = _request()
    finding = _finding(request)

    assert finding.scope is AttachmentSemanticFindingScopeV2.ARTIFACT
    assert finding.severity is AttachmentSemanticReviewSeverityV2.P1
    assert finding.non_waivable is False
    assert finding.artifact_repair_allowed is True

    values = finding.model_dump(mode="python")
    values["severity"] = AttachmentSemanticReviewSeverityV2.P3
    with pytest.raises(ValidationError, match="finding policy"):
        AttachmentSemanticReviewFindingV2.model_validate(values)


def test_leakage_finding_is_p0_non_waivable_and_not_repairable_in_round() -> None:
    request = _request(AttachmentSemanticReviewRoundV2.LEAKAGE_EXECUTABILITY)
    finding = _finding(
        request,
        AttachmentSemanticReviewFindingCodeV2.ANSWER_BEARING_CONTENT,
    )

    assert finding.severity is AttachmentSemanticReviewSeverityV2.P0
    assert finding.non_waivable is True
    assert finding.artifact_repair_allowed is False


def test_review_result_binds_clean_context_and_outcome_matrix() -> None:
    request = _request()
    accepted = _result(request)

    validate_attachment_semantic_review_result_identity(accepted)
    assert accepted.outcome is AttachmentSemanticReviewOutcomeV2.ACCEPTED
    assert accepted.clean_context_attestation.fresh_context is True
    assert accepted.clean_context_attestation.runtime_resume_used is False
    assert accepted.clean_context_attestation.hidden_reasoning_included is False

    finding = _finding(request)
    repair = _result(request, findings=(finding,))
    assert repair.outcome is AttachmentSemanticReviewOutcomeV2.REQUIRES_REPAIR

    values = accepted.model_dump(mode="python")
    values["findings"] = (finding,)
    with pytest.raises(ValidationError, match="ACCEPTED"):
        AttachmentSemanticReviewResultV2.model_validate(values)


def test_resolution_requires_changed_subjects_and_disposition_matrix() -> None:
    request = _request(AttachmentSemanticReviewRoundV2.REALISM_CONSISTENCY)
    old_subject = _ref("attachment-output", "old")
    current_subject = _ref(
        "attachment-output",
        "current",
        digest="b" * 64,
    )
    resolution = AttachmentSemanticFindingResolutionV2(
        resolution_id="attachment-semantic-finding-resolution://pending",
        semantic_review_request_ref=attachment_semantic_review_request_ref(request),
        prior_finding_ref=_ref(
            "attachment-semantic-review-finding",
            "coverage",
        ),
        old_subject_refs=(old_subject,),
        current_subject_refs=(current_subject,),
        repair_plan_ref=_ref("targeted-repair-plan", "coverage"),
        repair_result_refs=(_ref("attachment-repair-result", "coverage"),),
        deterministic_revalidation_ref=_ref(
            "revision-deterministic-validation",
            "current",
        ),
        disposition=(AttachmentSemanticResolutionDispositionV2.CONFIRMED_RESOLVED),
        successor_finding_ref=None,
        evidence_ref_ids=("semantic-evidence://realism/resolution",),
        resolution_sha256="0" * 64,
    )
    digest = attachment_semantic_finding_resolution_carried_sha256(resolution)
    resolution = resolution.model_copy(
        update={
            "resolution_id": (f"attachment-semantic-finding-resolution://sha256/{digest}"),
            "resolution_sha256": digest,
        }
    )

    assert attachment_semantic_finding_resolution_ref(resolution).object_sha256 == digest

    values = resolution.model_dump(mode="python")
    values["current_subject_refs"] = values["old_subject_refs"]
    with pytest.raises(ValidationError, match="changed subjects"):
        AttachmentSemanticFindingResolutionV2.model_validate(values)

    values = resolution.model_dump(mode="python")
    values["disposition"] = AttachmentSemanticResolutionDispositionV2.PERSISTS
    with pytest.raises(ValidationError, match="successor"):
        AttachmentSemanticFindingResolutionV2.model_validate(values)
