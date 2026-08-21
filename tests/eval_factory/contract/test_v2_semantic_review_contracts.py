from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from env_mock_agent.facade import FacadeObjectRef
from env_mock_agent.facade.semantic_review_v2 import (
    AttachmentSemanticFindingScopeV2,
    AttachmentSemanticReviewFindingCodeV2,
    AttachmentSemanticReviewFindingV2,
    AttachmentSemanticReviewRoundV2,
    attachment_semantic_finding_policy,
    attachment_semantic_review_finding_carried_sha256,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.review_v2 import (
    AttachmentCandidateRevisionV2,
    CandidateArtifactVersionV2,
    CoverageSolvabilityReviewViewV2,
    RevisionDeterministicValidationOutcomeV2,
    RevisionDeterministicValidationV2,
    SemanticFindingResolutionV2,
    SemanticFindingScopeV2,
    SemanticReviewerRoleV2,
    SemanticReviewFindingV2,
    SemanticReviewPolicyV2,
    SemanticReviewRoundV2,
    attachment_candidate_revision_carried_sha256,
    attachment_candidate_revision_ref,
    candidate_artifact_version_ref,
    revision_deterministic_validation_ref,
    semantic_finding_resolution_is_current,
    semantic_review_finding_is_current,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64


def _ref(
    object_type: str,
    suffix: str,
    *,
    digest: str = HASH,
    version: str = "v2",
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _facade_ref(ref: ObjectRef) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=ref.object_type,
        object_id=ref.object_id,
        object_version=ref.object_version,
        object_sha256=ref.object_sha256,
    )


def _audit(*refs: ObjectRef) -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 7, 29, tzinfo=UTC),
        created_by="semantic-review-contract-test",
        governing_versions=(VersionBinding(component="semantic-review", version="r5-09"),),
        input_refs=tuple(
            sorted(
                set(refs),
                key=lambda ref: (
                    ref.object_type,
                    ref.object_id,
                    ref.object_version,
                    ref.object_sha256,
                ),
            )
        ),
    )


def _artifact(
    artifact_id: str,
    *,
    digest: str = HASH,
) -> CandidateArtifactVersionV2:
    return CandidateArtifactVersionV2.create_base(
        artifact_id=artifact_id,
        attachment_dependency_id=(f"attachment-dependency://{artifact_id.rsplit('/', 1)[-1]}"),
        artifact_build_result_ref=_ref(
            "artifact-build-result",
            artifact_id.rsplit("/", 1)[-1],
        ),
        build_spec_ref=_ref(
            "artifact-build-spec",
            artifact_id.rsplit("/", 1)[-1],
        ),
        execution_request_ref=_ref(
            "attachment-execution-request",
            artifact_id.rsplit("/", 1)[-1],
        ),
        execution_result_ref=_ref(
            "attachment-execution-result",
            artifact_id.rsplit("/", 1)[-1],
        ),
        output_ref=_ref(
            "attachment-output",
            artifact_id.rsplit("/", 1)[-1],
            digest=digest,
        ),
        logical_path=f"inputs/{artifact_id.rsplit('/', 1)[-1]}.txt",
        media_type="text/plain",
        declared_validator_ids=("text-validator",),
        derivation_root_refs=(
            _ref("producer-task-view", "current"),
            _ref("artifact-build-spec", artifact_id.rsplit("/", 1)[-1]),
        ),
    )


def _revision() -> AttachmentCandidateRevisionV2:
    reconstruction_ref = _ref("attachment-reconstruction-result", "base")
    artifacts = (
        _artifact("artifact://a"),
        _artifact("artifact://b"),
    )
    return AttachmentCandidateRevisionV2.create_initial(
        base_reconstruction_result_ref=reconstruction_ref,
        artifact_versions=artifacts,
        audit=_audit(
            reconstruction_ref,
            *(candidate_artifact_version_ref(item) for item in artifacts),
        ),
    )


def _validation(
    revision: AttachmentCandidateRevisionV2,
) -> RevisionDeterministicValidationV2:
    revision_ref = attachment_candidate_revision_ref(revision)
    source_ref = _ref("deterministic-item-validation-result", "base")
    return RevisionDeterministicValidationV2.create(
        candidate_revision_ref=revision_ref,
        source_deterministic_validation_result_ref=source_ref,
        artifact_validation_result_refs=(
            _ref("artifact-deterministic-validation-result", "a"),
            _ref("artifact-deterministic-validation-result", "b"),
        ),
        finding_refs=(),
        output_refs=revision.candidate_output_refs,
        outcome=RevisionDeterministicValidationOutcomeV2.PASSED,
        audit=_audit(revision_ref, source_ref),
    )


def _policy() -> SemanticReviewPolicyV2:
    return SemanticReviewPolicyV2.create(
        model_profile_refs={
            SemanticReviewRoundV2.COVERAGE_SOLVABILITY: _ref(
                "model-profile",
                "coverage",
                version="v1",
            ),
            SemanticReviewRoundV2.REALISM_CONSISTENCY: _ref(
                "model-profile",
                "realism",
                version="v1",
            ),
            SemanticReviewRoundV2.LEAKAGE_EXECUTABILITY: _ref(
                "model-profile",
                "leakage",
                version="v1",
            ),
        },
        prompt_versions={
            SemanticReviewRoundV2.COVERAGE_SOLVABILITY: "coverage/v1",
            SemanticReviewRoundV2.REALISM_CONSISTENCY: "realism/v1",
            SemanticReviewRoundV2.LEAKAGE_EXECUTABILITY: "leakage/v1",
        },
        projection_policy_refs={
            SemanticReviewRoundV2.COVERAGE_SOLVABILITY: _ref(
                "projection-policy",
                "coverage",
                version="v1",
            ),
            SemanticReviewRoundV2.REALISM_CONSISTENCY: _ref(
                "projection-policy",
                "realism",
                version="v1",
            ),
            SemanticReviewRoundV2.LEAKAGE_EXECUTABILITY: _ref(
                "projection-policy",
                "leakage",
                version="v1",
            ),
        },
    )


def test_initial_candidate_revision_has_exact_pre_quality_boundary() -> None:
    revision = _revision()

    assert revision.revision == 1
    assert revision.predecessor_revision_ref is None
    assert revision.changed_artifact_ids == ()
    assert revision.artifact_version_refs == tuple(
        candidate_artifact_version_ref(item) for item in revision.artifact_versions
    )
    assert revision.accepted_artifact_refs == ()
    assert revision.environment_spec_ref is None
    assert revision.provenance_manifest_ref is None
    assert revision.quality_report_ref is None
    assert revision.package_sha256 is None
    assert revision.input_state_only is None
    assert attachment_candidate_revision_ref(revision).object_sha256 == (revision.candidate_revision_sha256)

    serialized = revision.model_dump(mode="python")
    serialized["quality_report_ref"] = _ref("quality-report", "premature")
    with pytest.raises(ValidationError, match="quality"):
        AttachmentCandidateRevisionV2.model_validate(serialized)


def test_candidate_revision_identity_is_order_and_audit_independent() -> None:
    first = _revision()
    reconstruction_ref = first.base_reconstruction_result_ref
    reversed_value = AttachmentCandidateRevisionV2.create_initial(
        base_reconstruction_result_ref=reconstruction_ref,
        artifact_versions=tuple(reversed(first.artifact_versions)),
        audit=_audit(
            reconstruction_ref,
            *(candidate_artifact_version_ref(item) for item in first.artifact_versions),
        ).model_copy(
            update={
                "created_at": datetime(2026, 7, 30, tzinfo=UTC),
                "created_by": "another-reviewer",
            }
        ),
    )

    assert attachment_candidate_revision_carried_sha256(first) == (
        attachment_candidate_revision_carried_sha256(reversed_value)
    )
    assert first.canonical_sha256() != reversed_value.canonical_sha256()


def test_revision_deterministic_validation_binds_exact_current_outputs() -> None:
    revision = _revision()
    validation = _validation(revision)

    assert validation.output_refs == revision.candidate_output_refs
    assert revision_deterministic_validation_ref(validation).object_sha256 == (validation.validation_sha256)

    values = validation.model_dump(mode="python")
    values["output_refs"] = (_ref("attachment-output", "stale", digest=OTHER_HASH),)
    with pytest.raises(ValidationError, match="identity"):
        RevisionDeterministicValidationV2.model_validate(values)


def test_coverage_view_has_closed_round_role_and_denied_families() -> None:
    revision = _revision()
    validation = _validation(revision)
    policy = _policy()
    revision_ref = attachment_candidate_revision_ref(revision)
    validation_ref = revision_deterministic_validation_ref(validation)
    prompt_ref = _ref("query-spec-public-view", "current")
    manifest_ref = _ref("candidate-package-inventory", "current")
    rubric_ref = _ref("public-rubric-requirements", "current")
    evidence_ref = _ref("evidence-bundle", "coverage", version="v1")

    view = CoverageSolvabilityReviewViewV2.create(
        candidate_revision_ref=revision_ref,
        deterministic_validation_result_ref=validation_ref,
        projection_policy_ref=policy.projection_for(SemanticReviewRoundV2.COVERAGE_SOLVABILITY),
        query_public_view_ref=prompt_ref,
        candidate_manifest_ref=manifest_ref,
        public_rubric_requirement_refs=(rubric_ref,),
        safe_evidence_bundle_refs=(evidence_ref,),
        deterministic_finding_refs=(),
        audit=_audit(
            revision_ref,
            validation_ref,
            prompt_ref,
            manifest_ref,
            rubric_ref,
            evidence_ref,
        ),
    )

    assert view.round is SemanticReviewRoundV2.COVERAGE_SOLVABILITY
    assert view.reviewer_role is (SemanticReviewerRoleV2.COVERAGE_SOLVABILITY_REVIEWER)
    assert "BUILD_TRANSCRIPT" in view.denied_data_families
    serialized = json.dumps(view.model_dump(mode="json"), sort_keys=True)
    for denied in (
        "private_reference",
        "grader_rule",
        "runtime_transcript",
        "hidden_reasoning",
    ):
        assert denied not in serialized

    values = view.model_dump(mode="python")
    values["private_reference_ref"] = _ref("private-reference", "forbidden")
    with pytest.raises(ValidationError, match="extra"):
        CoverageSolvabilityReviewViewV2.model_validate(values)


def test_finding_and_resolution_currentness_follow_exact_subject_hashes() -> None:
    revision = _revision()
    artifact = revision.artifact_versions[0]
    policy = attachment_semantic_finding_policy(
        AttachmentSemanticReviewFindingCodeV2.REQUIRED_COVERAGE_MISSING
    )
    subjects = tuple(
        sorted(
            (
                attachment_candidate_revision_ref(revision),
                candidate_artifact_version_ref(artifact),
                artifact.output_ref,
            ),
            key=lambda ref: (
                ref.object_type,
                ref.object_id,
                ref.object_version,
                ref.object_sha256,
            ),
        )
    )
    facade_finding = AttachmentSemanticReviewFindingV2(
        finding_id="attachment-semantic-review-finding://pending",
        semantic_review_request_ref=_facade_ref(_ref("attachment-semantic-review-request", "coverage")),
        round=AttachmentSemanticReviewRoundV2.COVERAGE_SOLVABILITY,
        scope=AttachmentSemanticFindingScopeV2.ARTIFACT,
        code=(AttachmentSemanticReviewFindingCodeV2.REQUIRED_COVERAGE_MISSING),
        candidate_revision_ref=_facade_ref(attachment_candidate_revision_ref(revision)),
        subject_refs=tuple(_facade_ref(ref) for ref in subjects),
        artifact_ids=(artifact.artifact_id,),
        evidence_ref_ids=("semantic-evidence://coverage/input",),
        predecessor_finding_ref=None,
        severity=policy.severity,
        non_waivable=policy.non_waivable,
        artifact_repair_allowed=policy.artifact_repair_allowed,
        finding_sha256="0" * 64,
    )
    digest = attachment_semantic_review_finding_carried_sha256(facade_finding)
    facade_finding = facade_finding.model_copy(
        update={
            "finding_id": (f"attachment-semantic-review-finding://sha256/{digest}"),
            "finding_sha256": digest,
        }
    )
    finding = SemanticReviewFindingV2.create(
        facade_finding=facade_finding,
        facade_review_result_ref=_ref(
            "attachment-semantic-review-result",
            "coverage",
        ),
        audit=_audit(),
    )

    changed_artifacts = (
        _artifact("artifact://a", digest=OTHER_HASH),
        revision.artifact_versions[1],
    )
    changed_revision = AttachmentCandidateRevisionV2.create_initial(
        base_reconstruction_result_ref=(revision.base_reconstruction_result_ref),
        artifact_versions=changed_artifacts,
        audit=_audit(
            revision.base_reconstruction_result_ref,
            *(candidate_artifact_version_ref(item) for item in changed_artifacts),
        ),
    )

    assert finding.scope is SemanticFindingScopeV2.ARTIFACT
    assert semantic_review_finding_is_current(finding, revision) is True
    assert semantic_review_finding_is_current(finding, changed_revision) is False

    resolution = SemanticFindingResolutionV2(
        semantic_finding_resolution_id="semantic-finding-resolution://pending",
        facade_resolution_ref=_ref(
            "attachment-semantic-finding-resolution",
            "coverage",
        ),
        resolving_stage_run_ref=_ref(
            "stage-run",
            "realism",
            version="identity/v1",
        ),
        prior_finding_ref=_ref("semantic-review-finding", "coverage"),
        old_subject_refs=finding.subject_refs,
        current_subject_refs=(attachment_candidate_revision_ref(revision),),
        repair_plan_ref=_ref("targeted-repair-plan", "coverage"),
        repair_result_refs=(_ref("repaired-artifact-build-result", "a"),),
        deterministic_revalidation_ref=_ref(
            "revision-deterministic-validation",
            "current",
        ),
        disposition="CONFIRMED_RESOLVED",
        successor_finding_ref=None,
        evidence_ref_ids=("semantic-evidence://realism/resolution",),
        resolution_sha256="0" * 64,
        audit=_audit(),
    )
    assert semantic_finding_resolution_is_current(resolution, revision) is True
    assert (
        semantic_finding_resolution_is_current(
            resolution,
            changed_revision,
        )
        is False
    )
