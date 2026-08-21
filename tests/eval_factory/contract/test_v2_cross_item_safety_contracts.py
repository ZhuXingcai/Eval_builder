from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from env_mock_agent.facade import (
    ATTACHMENT_CROSS_ITEM_SAFETY_POLICY_VERSION,
    AttachmentCrossItemSafetyScanFailureCodeV2,
    AttachmentCrossItemSafetyScanLimitsV2,
    AttachmentCrossItemSafetyScanRequestV2,
    AttachmentCrossItemSafetyScanResultV2,
    AttachmentCrossItemSafetyScanStatusV2,
    AttachmentValidationFingerprintCategoryV2,
    AttachmentValidationFingerprintMatchKindV2,
    AttachmentValidationFingerprintV2,
    FacadeObjectRef,
    attachment_cross_item_safety_scan_request_carried_sha256,
    attachment_cross_item_safety_scan_request_ref,
    attachment_cross_item_safety_scan_result_carried_sha256,
    attachment_cross_item_safety_scan_result_ref,
)
from env_mock_agent.facade.validation_v2 import (
    PROMPT_LEAKAGE_NORMALIZATION_VERSION,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.cross_item_safety_v2 import (
    AnswerReuseMatchKindV2,
    AnswerReusePairEvidenceV2,
    AttachmentCrossItemSafetyScanEvidenceV2,
    CrossItemSafetyClusterKindV2,
    CrossItemSafetyClusterV2,
    CrossItemSafetyEvidenceKindV2,
    CrossItemSafetyPolicyV2,
    CrossItemSafetyResultV2,
    CrossItemSafetySurfaceKindV2,
    CrossItemVisibleMatchEvidenceV2,
    answer_reuse_pair_evidence_v2_ref,
    attachment_cross_item_safety_scan_evidence_v2_ref,
    cross_item_safety_cluster_v2_ref,
    cross_item_safety_policy_v2_ref,
    cross_item_safety_result_v2_ref,
    cross_item_visible_match_evidence_v2_ref,
    validate_cross_item_safety_result_v2_identity,
)
from eval_factory.contracts.task_v2 import (
    PromptLeakageCategoryV2,
    PromptLeakageFingerprintV2,
    PromptLeakageMatchKindV2,
    prompt_leakage_fingerprint_carried_sha256,
    prompt_leakage_fingerprint_ref,
    validate_prompt_leakage_fingerprint_identity,
)

NOW = datetime(2026, 8, 1, tzinfo=UTC)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ref(
    object_type: str,
    suffix: str,
    *,
    digest: str | None = None,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r7-02/{suffix}",
        object_version="v2",
        object_sha256=digest or _digest(f"{object_type}:{suffix}"),
    )


def _facade_ref(value: ObjectRef) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=value.object_type,
        object_id=value.object_id,
        object_version=value.object_version,
        object_sha256=value.object_sha256,
    )


def _object_ref(value: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=value.object_type,
        object_id=value.object_id,
        object_version=value.object_version,
        object_sha256=value.object_sha256,
    )


def _audit(*refs: ObjectRef) -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="cross-item-contract-test",
        governing_versions=(
            VersionBinding(
                component="cross-item-safety",
                version="r7-02-v1",
            ),
        ),
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


def _limits() -> AttachmentCrossItemSafetyScanLimitsV2:
    return AttachmentCrossItemSafetyScanLimitsV2(
        max_file_bytes=10_000_000,
        max_extracted_characters=1_000_000,
        max_inventory_members=10_000,
        max_nested_depth=4,
        max_expanded_bytes=50_000_000,
        max_compression_ratio_milli=1_000_000,
        max_fingerprint_count=10_000,
        max_match_count=1_000,
    )


def _policy() -> CrossItemSafetyPolicyV2:
    return CrossItemSafetyPolicyV2.create(
        min_shared_answer_windows=2,
        min_answer_reuse_coverage_bps=8_000,
        max_items=100,
        max_attachments=1_000,
        max_total_fingerprints=100_000,
        max_foreign_fingerprints_per_target=10_000,
        max_prompt_characters=100_000,
        max_prompt_scan_comparisons=10_000_000,
        max_answer_source_pairs=1_000_000,
        max_answer_window_comparisons=10_000_000,
        max_visible_matches=100_000,
        max_reuse_pairs=100_000,
        max_clusters=100_000,
        attachment_scan_limits=_limits(),
        audit=_audit(),
    )


def _fingerprint(
    suffix: str,
    *,
    category: PromptLeakageCategoryV2,
    match_kind: PromptLeakageMatchKindV2,
    token_count: int,
    digest: str | None = None,
) -> PromptLeakageFingerprintV2:
    pending = PromptLeakageFingerprintV2(
        fingerprint_id="prompt-leakage-fingerprint://pending",
        category=category,
        source_subject_ref=_ref("restricted-source", suffix),
        source_provenance_decision_ref=_ref("provenance-decision", suffix),
        classification_evidence_ref=_ref("safety-classification", suffix),
        match_kind=match_kind,
        digest_sha256=digest or _digest(f"fingerprint:{suffix}"),
        token_count=token_count,
        normalization_version=PROMPT_LEAKAGE_NORMALIZATION_VERSION,
    )
    fingerprint_digest = prompt_leakage_fingerprint_carried_sha256(pending)
    return pending.model_copy(
        update={"fingerprint_id": (f"prompt-leakage-fingerprint://sha256/{fingerprint_digest}")}
    )


def _provider_request(
    fingerprint: PromptLeakageFingerprintV2,
) -> AttachmentCrossItemSafetyScanRequestV2:
    output_ref = _ref("attachment-output", "target")
    facade_fingerprint = AttachmentValidationFingerprintV2(
        fingerprint_id=fingerprint.fingerprint_id,
        category=AttachmentValidationFingerprintCategoryV2(fingerprint.category.value),
        match_kind=AttachmentValidationFingerprintMatchKindV2(fingerprint.match_kind.value),
        digest_sha256=fingerprint.digest_sha256,
        token_count=fingerprint.token_count,
        normalization_version=fingerprint.normalization_version,
    )
    pending = AttachmentCrossItemSafetyScanRequestV2(
        scan_request_id="attachment-cross-item-safety-scan-request://pending",
        target_item_id="dataset-item://r7-02/b",
        item_quality_result_ref=_facade_ref(_ref("item-quality-compilation-result", "b")),
        environment_artifact_ref=_facade_ref(_ref("environment-artifact", "b")),
        candidate_artifact_version_ref=_facade_ref(_ref("candidate-artifact-version", "b")),
        output_ref=_facade_ref(output_ref),
        artifact_validation_result_ref=_facade_ref(_ref("artifact-deterministic-validation-result", "b")),
        logical_path="inputs/source.txt",
        media_type="text/plain",
        content_sha256=output_ref.object_sha256,
        size_bytes=128,
        foreign_reference_set_refs=(_facade_ref(_ref("prompt-leakage-reference-set", "a")),),
        foreign_fingerprints=(facade_fingerprint,),
        limits=_limits(),
        normalization_version=PROMPT_LEAKAGE_NORMALIZATION_VERSION,
        policy_version=ATTACHMENT_CROSS_ITEM_SAFETY_POLICY_VERSION,
        idempotency_key="attachment-cross-item-safety-idempotency://contract",
        request_sha256="0" * 64,
    )
    digest = attachment_cross_item_safety_scan_request_carried_sha256(pending)
    return pending.model_copy(
        update={
            "scan_request_id": (f"attachment-cross-item-safety-scan-request://sha256/{digest}"),
            "request_sha256": digest,
        }
    )


def _provider_result(
    request: AttachmentCrossItemSafetyScanRequestV2,
) -> AttachmentCrossItemSafetyScanResultV2:
    pending = AttachmentCrossItemSafetyScanResultV2(
        scan_result_id="attachment-cross-item-safety-scan-result://pending",
        scan_request_ref=attachment_cross_item_safety_scan_request_ref(request),
        environment_artifact_ref=request.environment_artifact_ref,
        output_ref=request.output_ref,
        output_sha256=request.content_sha256,
        status=AttachmentCrossItemSafetyScanStatusV2.PASSED,
        matched_fingerprint_ids=(request.foreign_fingerprints[0].fingerprint_id,),
        scanned_member_count=1,
        scan_complete=True,
        failure_code=None,
        extractor_version="attachment-cross-item-safety-extractor/r7-02-v1",
        normalization_version=request.normalization_version,
        policy_version=request.policy_version,
        result_sha256="0" * 64,
    )
    digest = attachment_cross_item_safety_scan_result_carried_sha256(pending)
    return pending.model_copy(
        update={
            "scan_result_id": (f"attachment-cross-item-safety-scan-result://sha256/{digest}"),
            "result_sha256": digest,
        }
    )


def _scan_evidence(
    policy: CrossItemSafetyPolicyV2,
    fingerprint: PromptLeakageFingerprintV2,
) -> AttachmentCrossItemSafetyScanEvidenceV2:
    request = _provider_request(fingerprint)
    result = _provider_result(request)
    refs = (
        _object_ref(request.item_quality_result_ref),
        _object_ref(request.environment_artifact_ref),
        _object_ref(request.candidate_artifact_version_ref),
        _object_ref(request.output_ref),
        _object_ref(request.artifact_validation_result_ref),
        _object_ref(attachment_cross_item_safety_scan_request_ref(request)),
        _object_ref(attachment_cross_item_safety_scan_result_ref(result)),
        _object_ref(request.foreign_reference_set_refs[0]),
        cross_item_safety_policy_v2_ref(policy),
    )
    return AttachmentCrossItemSafetyScanEvidenceV2.create(
        target_item_id=request.target_item_id,
        item_quality_result_ref=request.item_quality_result_ref,
        environment_artifact_ref=request.environment_artifact_ref,
        candidate_artifact_version_ref=request.candidate_artifact_version_ref,
        output_ref=request.output_ref,
        artifact_validation_result_ref=request.artifact_validation_result_ref,
        facade_request_ref=attachment_cross_item_safety_scan_request_ref(request),
        facade_result=result,
        facade_result_ref=attachment_cross_item_safety_scan_result_ref(result),
        foreign_reference_set_refs=request.foreign_reference_set_refs,
        policy_ref=cross_item_safety_policy_v2_ref(policy),
        audit=_audit(*refs),
    )


def _visible_match(
    policy: CrossItemSafetyPolicyV2,
    fingerprint: PromptLeakageFingerprintV2,
) -> CrossItemVisibleMatchEvidenceV2:
    refs = (
        _ref("prompt-leakage-reference-set", "a"),
        fingerprint.source_subject_ref,
        _ref("task-draft", "b"),
        prompt_leakage_fingerprint_ref(fingerprint),
        cross_item_safety_policy_v2_ref(policy),
    )
    return CrossItemVisibleMatchEvidenceV2.create(
        evidence_kind=CrossItemSafetyEvidenceKindV2.LEAKAGE,
        target_surface=CrossItemSafetySurfaceKindV2.PROMPT,
        source_item_id="dataset-item://r7-02/a",
        target_item_id="dataset-item://r7-02/b",
        source_reference_set_ref=refs[0],
        source_subject_ref=refs[1],
        target_subject_ref=refs[2],
        category=fingerprint.category,
        matched_fingerprint_refs=(refs[3],),
        match_kinds=(fingerprint.match_kind,),
        attachment_scan_evidence_ref=None,
        policy_ref=refs[4],
        audit=_audit(*refs),
    )


def _reuse_pair(
    policy: CrossItemSafetyPolicyV2,
    left: PromptLeakageFingerprintV2,
    right: PromptLeakageFingerprintV2,
) -> AnswerReusePairEvidenceV2:
    refs = (
        _ref("prompt-leakage-reference-set", "a"),
        _ref("prompt-leakage-reference-set", "b"),
        left.source_subject_ref,
        right.source_subject_ref,
        prompt_leakage_fingerprint_ref(left),
        prompt_leakage_fingerprint_ref(right),
        cross_item_safety_policy_v2_ref(policy),
    )
    return AnswerReusePairEvidenceV2.create(
        left_item_id="dataset-item://r7-02/a",
        right_item_id="dataset-item://r7-02/b",
        left_reference_set_ref=refs[0],
        right_reference_set_ref=refs[1],
        left_source_subject_ref=refs[2],
        right_source_subject_ref=refs[3],
        left_matched_fingerprint_refs=(refs[4],),
        right_matched_fingerprint_refs=(refs[5],),
        match_kind=AnswerReuseMatchKindV2.FULL_SOURCE,
        shared_window_count=0,
        shorter_source_window_count=0,
        coverage_bps=10_000,
        policy_ref=refs[6],
        audit=_audit(*refs),
    )


def test_policy_and_existing_fingerprint_have_recomputable_identity() -> None:
    policy = _policy()
    fingerprint = _fingerprint(
        "private",
        category=PromptLeakageCategoryV2.PRIVATE_REFERENCE,
        match_kind=PromptLeakageMatchKindV2.TOKEN_WINDOW,
        token_count=8,
    )

    assert cross_item_safety_policy_v2_ref(policy).object_sha256 == policy.policy_sha256
    validate_prompt_leakage_fingerprint_identity(fingerprint)
    assert prompt_leakage_fingerprint_ref(fingerprint).object_id == (fingerprint.fingerprint_id)

    with pytest.raises(ValueError, match="fingerprint identity"):
        validate_prompt_leakage_fingerprint_identity(fingerprint.model_copy(update={"token_count": 7}))
    with pytest.raises(ValidationError, match="Extra inputs"):
        CrossItemSafetyPolicyV2.model_validate(
            {
                **policy.model_dump(mode="python"),
                "semantic_similarity_threshold": 0.8,
            }
        )


def test_attachment_scan_evidence_binds_passed_provider_result() -> None:
    policy = _policy()
    fingerprint = _fingerprint(
        "answer",
        category=PromptLeakageCategoryV2.FINAL_ANSWER,
        match_kind=PromptLeakageMatchKindV2.TOKEN_WINDOW,
        token_count=8,
    )
    evidence = _scan_evidence(policy, fingerprint)

    assert (
        attachment_cross_item_safety_scan_evidence_v2_ref(evidence).object_sha256 == evidence.evidence_sha256
    )
    blocked = evidence.facade_result.model_copy(
        update={
            "status": AttachmentCrossItemSafetyScanStatusV2.BLOCKED,
            "matched_fingerprint_ids": (),
            "scanned_member_count": 0,
            "scan_complete": False,
            "failure_code": (AttachmentCrossItemSafetyScanFailureCodeV2.CONTENT_UNSCANNABLE),
            "extractor_version": None,
        }
    )
    with pytest.raises(ValidationError):
        AttachmentCrossItemSafetyScanEvidenceV2.model_validate(
            {
                **evidence.model_dump(mode="python"),
                "facade_result": blocked,
            }
        )


def test_visible_match_is_directed_non_waivable_and_category_exact() -> None:
    policy = _policy()
    fingerprint = _fingerprint(
        "private",
        category=PromptLeakageCategoryV2.PRIVATE_REFERENCE,
        match_kind=PromptLeakageMatchKindV2.TOKEN_WINDOW,
        token_count=8,
    )
    evidence = _visible_match(policy, fingerprint)

    assert evidence.non_waivable is True
    assert cross_item_visible_match_evidence_v2_ref(evidence).object_sha256 == evidence.evidence_sha256
    with pytest.raises(ValidationError, match="distinct Items"):
        CrossItemVisibleMatchEvidenceV2.model_validate(
            {
                **evidence.model_dump(mode="python"),
                "target_item_id": evidence.source_item_id,
            }
        )
    with pytest.raises(ValidationError, match="category"):
        CrossItemVisibleMatchEvidenceV2.model_validate(
            {
                **evidence.model_dump(mode="python"),
                "evidence_kind": CrossItemSafetyEvidenceKindV2.CONTAMINATION,
            }
        )


def test_answer_reuse_pair_enforces_full_and_partial_matrices() -> None:
    policy = _policy()
    left = _fingerprint(
        "left",
        category=PromptLeakageCategoryV2.FINAL_ANSWER,
        match_kind=PromptLeakageMatchKindV2.NORMALIZED_FULL_TEXT,
        token_count=12,
        digest="c" * 64,
    )
    right = _fingerprint(
        "right",
        category=PromptLeakageCategoryV2.PRIVATE_REFERENCE,
        match_kind=PromptLeakageMatchKindV2.NORMALIZED_FULL_TEXT,
        token_count=12,
        digest="c" * 64,
    )
    pair = _reuse_pair(policy, left, right)

    assert pair.non_waivable is True
    assert answer_reuse_pair_evidence_v2_ref(pair).object_sha256 == pair.pair_sha256
    with pytest.raises(ValidationError, match="FULL_SOURCE"):
        AnswerReusePairEvidenceV2.model_validate(
            {
                **pair.model_dump(mode="python"),
                "coverage_bps": 9_999,
            }
        )
    with pytest.raises(ValidationError, match="PARTIAL_WINDOWS"):
        AnswerReusePairEvidenceV2.model_validate(
            {
                **pair.model_dump(mode="python"),
                "match_kind": AnswerReuseMatchKindV2.PARTIAL_WINDOWS,
                "shared_window_count": 1,
                "shorter_source_window_count": 2,
                "coverage_bps": 5_000,
            }
        )


def test_result_proves_direct_edges_and_exact_connected_components() -> None:
    policy = _policy()
    visible_fingerprint = _fingerprint(
        "private",
        category=PromptLeakageCategoryV2.PRIVATE_REFERENCE,
        match_kind=PromptLeakageMatchKindV2.TOKEN_WINDOW,
        token_count=8,
    )
    left = _fingerprint(
        "left-answer",
        category=PromptLeakageCategoryV2.FINAL_ANSWER,
        match_kind=PromptLeakageMatchKindV2.NORMALIZED_FULL_TEXT,
        token_count=12,
        digest="d" * 64,
    )
    right = _fingerprint(
        "right-answer",
        category=PromptLeakageCategoryV2.PRIVATE_REFERENCE,
        match_kind=PromptLeakageMatchKindV2.NORMALIZED_FULL_TEXT,
        token_count=12,
        digest="d" * 64,
    )
    visible = _visible_match(policy, visible_fingerprint)
    pair = _reuse_pair(policy, left, right)
    leakage_cluster = CrossItemSafetyClusterV2.create(
        cluster_kind=CrossItemSafetyClusterKindV2.LEAKAGE,
        member_item_ids=(
            "dataset-item://r7-02/a",
            "dataset-item://r7-02/b",
        ),
        direct_evidence_refs=(cross_item_visible_match_evidence_v2_ref(visible),),
        policy_ref=cross_item_safety_policy_v2_ref(policy),
        audit=_audit(
            cross_item_visible_match_evidence_v2_ref(visible),
            cross_item_safety_policy_v2_ref(policy),
        ),
    )
    reuse_cluster = CrossItemSafetyClusterV2.create(
        cluster_kind=CrossItemSafetyClusterKindV2.ANSWER_REUSE,
        member_item_ids=(
            "dataset-item://r7-02/a",
            "dataset-item://r7-02/b",
        ),
        direct_evidence_refs=(answer_reuse_pair_evidence_v2_ref(pair),),
        policy_ref=cross_item_safety_policy_v2_ref(policy),
        audit=_audit(
            answer_reuse_pair_evidence_v2_ref(pair),
            cross_item_safety_policy_v2_ref(policy),
        ),
    )
    graph_ref = _ref("resolved-job-work-graph", "batch")
    quality_refs = (
        _ref("item-quality-compilation-result", "a"),
        _ref("item-quality-compilation-result", "b"),
    )
    task_refs = (
        _ref("task-draft", "a"),
        _ref("task-draft", "b"),
    )
    reference_refs = (
        _ref("prompt-leakage-reference-set", "a"),
        _ref("prompt-leakage-reference-set", "b"),
    )
    answer_subjects = (
        left.source_subject_ref,
        right.source_subject_ref,
    )
    result = CrossItemSafetyResultV2.create(
        resolved_job_work_graph_ref=graph_ref,
        policy_ref=cross_item_safety_policy_v2_ref(policy),
        item_ids=(
            "dataset-item://r7-02/a",
            "dataset-item://r7-02/b",
        ),
        task_draft_refs=task_refs,
        leakage_reference_set_refs=reference_refs,
        item_quality_result_refs=quality_refs,
        answer_source_item_ids=(
            "dataset-item://r7-02/a",
            "dataset-item://r7-02/b",
        ),
        answer_source_subject_refs=answer_subjects,
        attachment_scan_evidence=(),
        visible_matches=(visible,),
        answer_reuse_pairs=(pair,),
        clusters=(leakage_cluster, reuse_cluster),
        evaluated_prompt_foreign_set_count=2,
        evaluated_attachment_foreign_set_count=0,
        evaluated_answer_source_pair_count=1,
        audit=_audit(
            graph_ref,
            cross_item_safety_policy_v2_ref(policy),
            *task_refs,
            *reference_refs,
            *quality_refs,
            *answer_subjects,
            cross_item_visible_match_evidence_v2_ref(visible),
            answer_reuse_pair_evidence_v2_ref(pair),
            cross_item_safety_cluster_v2_ref(leakage_cluster),
            cross_item_safety_cluster_v2_ref(reuse_cluster),
        ),
    )

    validate_cross_item_safety_result_v2_identity(result)
    assert cross_item_safety_result_v2_ref(result).object_sha256 == result.result_sha256
    serialized = json.dumps(result.model_dump(mode="json"), sort_keys=True)
    for denied in (
        "prompt_text",
        "source_text",
        "matched_text",
        "physical_path",
        "member_path",
        "token_inventory",
        "window_inventory",
        "provider_payload",
    ):
        assert denied not in serialized

    with pytest.raises(ValidationError, match="connected components"):
        CrossItemSafetyResultV2.model_validate(
            {
                **result.model_dump(mode="python"),
                "clusters": (reuse_cluster,),
                "cluster_count": 1,
            }
        )
