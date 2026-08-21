from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from env_mock_agent.facade import (
    ATTACHMENT_DUPLICATE_FINGERPRINT_POLICY_VERSION,
    ATTACHMENT_DUPLICATE_NORMALIZATION_VERSION,
    AttachmentDuplicateFingerprintLimitsV2,
    AttachmentDuplicateFingerprintOutcomeV2,
    AttachmentDuplicateFingerprintRequestV2,
    AttachmentDuplicateFingerprintResultV2,
    FacadeObjectRef,
    attachment_duplicate_fingerprint_request_carried_sha256,
    attachment_duplicate_fingerprint_request_ref,
    attachment_duplicate_fingerprint_result_carried_sha256,
)
from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.duplicate_v2 import (
    DUPLICATE_DETECTION_POLICY_VERSION,
    TASK_DUPLICATE_NORMALIZATION_VERSION,
    AttachmentDuplicateFingerprintV2,
    DuplicateClusterV2,
    DuplicateDetectionPolicyV2,
    DuplicateDetectionResultV2,
    DuplicateMatchKindV2,
    DuplicatePairEvidenceV2,
    DuplicateSubjectKindV2,
    TaskDuplicateFingerprintV2,
    attachment_duplicate_fingerprint_v2_ref,
    duplicate_cluster_v2_ref,
    duplicate_detection_policy_v2_ref,
    duplicate_detection_result_v2_ref,
    duplicate_pair_evidence_v2_ref,
    task_duplicate_fingerprint_v2_ref,
)

HASH = "a" * 64


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v2",
    digest: str = HASH,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r7-01/{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _facade_ref(
    object_type: str,
    suffix: str,
    *,
    digest: str = HASH,
) -> FacadeObjectRef:
    return FacadeObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r7-01/{suffix}",
        object_version="v2",
        object_sha256=digest,
    )


def _audit(
    *refs: ObjectRef,
    actor: str = "duplicate-contract-test",
    created_at: datetime | None = None,
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at or datetime(2026, 8, 1, tzinfo=UTC),
        created_by=actor,
        governing_versions=(
            VersionBinding(
                component="duplicate-detection",
                version="r7-01-v1",
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


def _limits() -> AttachmentDuplicateFingerprintLimitsV2:
    return AttachmentDuplicateFingerprintLimitsV2(
        max_file_bytes=10_000_000,
        max_extracted_characters=1_000_000,
        max_inventory_members=10_000,
        max_nested_depth=4,
        max_expanded_bytes=50_000_000,
        max_compression_ratio_milli=1_000_000,
        min_token_count=4,
        shingle_size=3,
        fingerprint_bits=256,
    )


def _policy(
    *,
    audit: ContractAudit | None = None,
) -> DuplicateDetectionPolicyV2:
    return DuplicateDetectionPolicyV2.create(
        task_near_threshold_bps=8_500,
        attachment_near_threshold_bps=8_750,
        task_shingle_size=3,
        attachment_shingle_size=3,
        fingerprint_bits=256,
        min_task_tokens=4,
        min_attachment_tokens=4,
        max_task_characters=100_000,
        max_attachment_characters=1_000_000,
        max_items=100,
        max_attachments=1_000,
        max_pair_comparisons=100_000,
        attachment_fingerprint_limits=_limits(),
        audit=audit or _audit(),
    )


def _task_fingerprint(
    *,
    item_id: str,
    suffix: str,
    policy: DuplicateDetectionPolicyV2,
    exact_digest: str = HASH,
    similarity_digest: str = "5" * 64,
    audit: ContractAudit | None = None,
) -> TaskDuplicateFingerprintV2:
    task_ref = _ref("task-draft", suffix)
    contract_ref = _ref("r4-task-contract-set", suffix)
    policy_ref = duplicate_detection_policy_v2_ref(policy)
    return TaskDuplicateFingerprintV2.create(
        item_id=item_id,
        task_draft_ref=task_ref,
        r4_task_contract_set_ref=contract_ref,
        exact_task_sha256=exact_digest,
        similarity_fingerprint=similarity_digest,
        token_count=12,
        shingle_count=10,
        policy_ref=policy_ref,
        audit=audit or _audit(task_ref, contract_ref, policy_ref),
    )


def _provider_values(
    *,
    suffix: str,
    supported: bool = True,
) -> tuple[
    AttachmentDuplicateFingerprintRequestV2,
    AttachmentDuplicateFingerprintResultV2,
]:
    output_ref = _facade_ref("attachment-output", suffix)
    pending_request = AttachmentDuplicateFingerprintRequestV2(
        fingerprint_request_id="attachment-duplicate-fingerprint-request://pending",
        item_quality_result_ref=_facade_ref(
            "item-quality-compilation-result",
            suffix,
        ),
        environment_artifact_ref=_facade_ref(
            "environment-artifact",
            suffix,
        ),
        candidate_artifact_version_ref=_facade_ref(
            "candidate-artifact-version",
            suffix,
        ),
        output_ref=output_ref,
        artifact_validation_result_ref=_facade_ref(
            "artifact-deterministic-validation-result",
            suffix,
        ),
        logical_path=f"inputs/{suffix}.txt",
        media_type="text/plain",
        content_sha256=HASH,
        size_bytes=128,
        limits=_limits(),
        normalization_version=ATTACHMENT_DUPLICATE_NORMALIZATION_VERSION,
        policy_version=ATTACHMENT_DUPLICATE_FINGERPRINT_POLICY_VERSION,
        idempotency_key=(f"attachment-duplicate-fingerprint-idempotency://{suffix}"),
        request_sha256="0" * 64,
    )
    request_digest = attachment_duplicate_fingerprint_request_carried_sha256(pending_request)
    request = pending_request.model_copy(
        update={
            "fingerprint_request_id": (f"attachment-duplicate-fingerprint-request://sha256/{request_digest}"),
            "request_sha256": request_digest,
        }
    )
    pending_result = AttachmentDuplicateFingerprintResultV2(
        fingerprint_result_id="attachment-duplicate-fingerprint-result://pending",
        fingerprint_request_ref=(attachment_duplicate_fingerprint_request_ref(request)),
        environment_artifact_ref=request.environment_artifact_ref,
        output_ref=request.output_ref,
        exact_content_sha256=request.content_sha256,
        outcome=(
            AttachmentDuplicateFingerprintOutcomeV2.SUPPORTED
            if supported
            else AttachmentDuplicateFingerprintOutcomeV2.UNSUPPORTED
        ),
        similarity_fingerprint="6" * 64 if supported else None,
        token_count=12 if supported else 0,
        shingle_count=10 if supported else 0,
        failure_code=(None if supported else "MEDIA_UNSUPPORTED"),
        extractor_version=("attachment-duplicate-extractor/r7-01-v1" if supported else None),
        normalization_version=ATTACHMENT_DUPLICATE_NORMALIZATION_VERSION,
        policy_version=ATTACHMENT_DUPLICATE_FINGERPRINT_POLICY_VERSION,
        result_sha256="0" * 64,
    )
    result_digest = attachment_duplicate_fingerprint_result_carried_sha256(pending_result)
    result = pending_result.model_copy(
        update={
            "fingerprint_result_id": (f"attachment-duplicate-fingerprint-result://sha256/{result_digest}"),
            "result_sha256": result_digest,
        }
    )
    return request, result


def _attachment_fingerprint(
    *,
    item_id: str,
    suffix: str,
    policy: DuplicateDetectionPolicyV2,
    supported: bool = True,
) -> AttachmentDuplicateFingerprintV2:
    request, result = _provider_values(
        suffix=suffix,
        supported=supported,
    )
    request_ref = _ref(
        "attachment-duplicate-fingerprint-request",
        suffix,
        digest=request.request_sha256,
    ).model_copy(update={"object_id": request.fingerprint_request_id})
    result_ref = _ref(
        "attachment-duplicate-fingerprint-result",
        suffix,
        digest=result.result_sha256,
    ).model_copy(update={"object_id": result.fingerprint_result_id})
    item_quality_ref = _ref(
        "item-quality-compilation-result",
        suffix,
    )
    environment_ref = _ref("environment-artifact", suffix)
    candidate_ref = _ref("candidate-artifact-version", suffix)
    output_ref = _ref("attachment-output", suffix)
    validation_ref = _ref(
        "artifact-deterministic-validation-result",
        suffix,
    )
    policy_ref = duplicate_detection_policy_v2_ref(policy)
    return AttachmentDuplicateFingerprintV2.create(
        item_id=item_id,
        item_quality_result_ref=item_quality_ref,
        environment_artifact_ref=environment_ref,
        candidate_artifact_version_ref=candidate_ref,
        output_ref=output_ref,
        artifact_validation_result_ref=validation_ref,
        facade_request_ref=request_ref,
        facade_result=result,
        facade_result_ref=result_ref,
        policy_ref=policy_ref,
        audit=_audit(
            item_quality_ref,
            environment_ref,
            candidate_ref,
            output_ref,
            validation_ref,
            request_ref,
            result_ref,
            policy_ref,
        ),
    )


def test_policy_is_explicit_strict_and_audit_independent() -> None:
    first = _policy()
    second = _policy(
        audit=_audit(
            actor="other-compiler",
            created_at=datetime(2026, 8, 2, tzinfo=UTC),
        )
    )

    assert first.policy_version == DUPLICATE_DETECTION_POLICY_VERSION
    assert first.clustering_mode == "CONNECTED_COMPONENTS"
    assert duplicate_detection_policy_v2_ref(first) == (duplicate_detection_policy_v2_ref(second))
    assert first.canonical_sha256() != second.canonical_sha256()
    values = first.model_dump(mode="python")
    values["embedding_model"] = "forbidden"
    with pytest.raises(ValidationError, match="extra"):
        DuplicateDetectionPolicyV2.model_validate(values)


def test_task_and_attachment_fingerprints_bind_exact_current_subjects() -> None:
    policy = _policy()
    task = _task_fingerprint(
        item_id="item://one",
        suffix="one",
        policy=policy,
    )
    attachment = _attachment_fingerprint(
        item_id="item://one",
        suffix="one",
        policy=policy,
    )

    assert task.normalization_version == TASK_DUPLICATE_NORMALIZATION_VERSION
    assert task_duplicate_fingerprint_v2_ref(task).object_sha256 == (task.fingerprint_sha256)
    assert attachment_duplicate_fingerprint_v2_ref(attachment).object_sha256 == attachment.fingerprint_sha256
    assert attachment.facade_result_ref.object_sha256 == (attachment.facade_result.result_sha256)


def test_pair_requires_distinct_items_canonical_order_and_match_matrix() -> None:
    policy = _policy()
    left = _task_fingerprint(
        item_id="item://left",
        suffix="left",
        policy=policy,
    )
    right = _task_fingerprint(
        item_id="item://right",
        suffix="right",
        policy=policy,
    )
    policy_ref = duplicate_detection_policy_v2_ref(policy)
    exact = DuplicatePairEvidenceV2.create(
        subject_kind=DuplicateSubjectKindV2.TASK,
        left_item_id=left.item_id,
        right_item_id=right.item_id,
        left_subject_ref=left.task_draft_ref,
        right_subject_ref=right.task_draft_ref,
        left_fingerprint_ref=task_duplicate_fingerprint_v2_ref(left),
        right_fingerprint_ref=task_duplicate_fingerprint_v2_ref(right),
        match_kind=DuplicateMatchKindV2.EXACT,
        similarity_bps=10_000,
        policy_ref=policy_ref,
        audit=_audit(
            left.task_draft_ref,
            right.task_draft_ref,
            task_duplicate_fingerprint_v2_ref(left),
            task_duplicate_fingerprint_v2_ref(right),
            policy_ref,
        ),
    )

    assert duplicate_pair_evidence_v2_ref(exact).object_sha256 == (exact.pair_sha256)
    values = exact.model_dump(mode="python")
    values["right_item_id"] = exact.left_item_id
    with pytest.raises(ValidationError, match="distinct Items"):
        DuplicatePairEvidenceV2.model_validate(values)
    values = exact.model_dump(mode="python")
    values.update(
        similarity_bps=9_999,
    )
    with pytest.raises(ValidationError, match="EXACT"):
        DuplicatePairEvidenceV2.model_validate(values)


def test_connected_component_preserves_only_observed_direct_edges() -> None:
    policy = _policy()
    task_refs = (
        _ref("task-draft", "a"),
        _ref("task-draft", "b"),
        _ref("task-draft", "c"),
    )
    pair_refs = (
        _ref("duplicate-pair-evidence", "a-b"),
        _ref("duplicate-pair-evidence", "b-c"),
    )
    policy_ref = duplicate_detection_policy_v2_ref(policy)
    cluster = DuplicateClusterV2.create(
        subject_kind=DuplicateSubjectKindV2.TASK,
        member_item_ids=(
            "item://a",
            "item://b",
            "item://c",
        ),
        member_subject_refs=task_refs,
        direct_pair_refs=pair_refs,
        exact_pair_count=1,
        near_pair_count=1,
        policy_ref=policy_ref,
        audit=_audit(
            *task_refs,
            *pair_refs,
            policy_ref,
        ),
    )

    assert len(cluster.member_subject_refs) == 3
    assert len(cluster.direct_pair_refs) == 2
    assert duplicate_cluster_v2_ref(cluster).object_sha256 == (cluster.cluster_sha256)


def test_result_derives_counts_and_unsupported_inventory_without_overclaim() -> None:
    policy = _policy()
    left = _task_fingerprint(
        item_id="item://left",
        suffix="left",
        policy=policy,
    )
    right = _task_fingerprint(
        item_id="item://right",
        suffix="right",
        policy=policy,
    )
    supported = _attachment_fingerprint(
        item_id="item://left",
        suffix="left",
        policy=policy,
    )
    unsupported = _attachment_fingerprint(
        item_id="item://right",
        suffix="right",
        policy=policy,
        supported=False,
    )
    policy_ref = duplicate_detection_policy_v2_ref(policy)
    graph_ref = _ref("resolved-job-work-graph", "current")
    item_quality_refs = (
        supported.item_quality_result_ref,
        unsupported.item_quality_result_ref,
    )
    result = DuplicateDetectionResultV2.create(
        resolved_job_work_graph_ref=graph_ref,
        policy_ref=policy_ref,
        item_quality_result_refs=item_quality_refs,
        task_fingerprints=(left, right),
        attachment_fingerprints=(supported, unsupported),
        duplicate_pairs=(),
        duplicate_clusters=(),
        audit=_audit(
            graph_ref,
            policy_ref,
            *item_quality_refs,
            task_duplicate_fingerprint_v2_ref(left),
            task_duplicate_fingerprint_v2_ref(right),
            attachment_duplicate_fingerprint_v2_ref(supported),
            attachment_duplicate_fingerprint_v2_ref(unsupported),
            unsupported.environment_artifact_ref,
        ),
    )

    assert result.evaluated_task_pair_count == 1
    assert result.evaluated_attachment_pair_count == 1
    assert result.unsupported_attachment_subject_refs == (unsupported.environment_artifact_ref,)
    assert result.unsupported_attachment_count == 1
    assert duplicate_detection_result_v2_ref(result).object_sha256 == (result.result_sha256)
    serialized = json.dumps(result.model_dump(mode="json"), sort_keys=True)
    for denied in (
        "prompt",
        "extracted_text",
        "token_inventory",
        "rubric",
        "reference",
        "release_decision",
        "batch_quality_report",
    ):
        assert denied not in serialized.casefold()


def test_result_requires_exact_connected_component_cluster_coverage() -> None:
    policy = _policy()
    policy_ref = duplicate_detection_policy_v2_ref(policy)
    tasks = tuple(
        _task_fingerprint(
            item_id=f"item://{suffix}",
            suffix=suffix,
            policy=policy,
            exact_digest=character * 64,
            similarity_digest=character * 64,
        )
        for suffix, character in (
            ("a", "1"),
            ("b", "2"),
            ("c", "3"),
        )
    )

    def pair(
        left: TaskDuplicateFingerprintV2,
        right: TaskDuplicateFingerprintV2,
        kind: DuplicateMatchKindV2,
    ) -> DuplicatePairEvidenceV2:
        similarity = 10_000 if kind is DuplicateMatchKindV2.EXACT else 9_500
        return DuplicatePairEvidenceV2.create(
            subject_kind=DuplicateSubjectKindV2.TASK,
            left_item_id=left.item_id,
            right_item_id=right.item_id,
            left_subject_ref=left.task_draft_ref,
            right_subject_ref=right.task_draft_ref,
            left_fingerprint_ref=task_duplicate_fingerprint_v2_ref(left),
            right_fingerprint_ref=task_duplicate_fingerprint_v2_ref(right),
            match_kind=kind,
            similarity_bps=similarity,
            policy_ref=policy_ref,
            audit=_audit(
                left.task_draft_ref,
                right.task_draft_ref,
                task_duplicate_fingerprint_v2_ref(left),
                task_duplicate_fingerprint_v2_ref(right),
                policy_ref,
            ),
        )

    pairs = (
        pair(tasks[0], tasks[1], DuplicateMatchKindV2.EXACT),
        pair(tasks[1], tasks[2], DuplicateMatchKindV2.NEAR),
    )
    pair_refs = tuple(duplicate_pair_evidence_v2_ref(item) for item in pairs)
    member_refs = tuple(item.task_draft_ref for item in tasks)
    cluster = DuplicateClusterV2.create(
        subject_kind=DuplicateSubjectKindV2.TASK,
        member_item_ids=tuple(item.item_id for item in tasks),
        member_subject_refs=member_refs,
        direct_pair_refs=pair_refs,
        exact_pair_count=1,
        near_pair_count=1,
        policy_ref=policy_ref,
        audit=_audit(*member_refs, *pair_refs, policy_ref),
    )
    graph_ref = _ref("resolved-job-work-graph", "connected")
    item_quality_refs = tuple(_ref("item-quality-compilation-result", suffix) for suffix in ("a", "b", "c"))
    result = DuplicateDetectionResultV2.create(
        resolved_job_work_graph_ref=graph_ref,
        policy_ref=policy_ref,
        item_quality_result_refs=item_quality_refs,
        task_fingerprints=tasks,
        attachment_fingerprints=(),
        duplicate_pairs=pairs,
        duplicate_clusters=(cluster,),
        audit=_audit(
            graph_ref,
            policy_ref,
            *item_quality_refs,
            *(task_duplicate_fingerprint_v2_ref(item) for item in tasks),
            *pair_refs,
            duplicate_cluster_v2_ref(cluster),
        ),
    )

    with pytest.raises(ValidationError, match="exactly cover"):
        DuplicateDetectionResultV2.model_validate(
            {
                **result.model_dump(mode="python"),
                "duplicate_clusters": (),
            }
        )

    disconnected = DuplicateClusterV2.create(
        subject_kind=DuplicateSubjectKindV2.TASK,
        member_item_ids=tuple(item.item_id for item in tasks),
        member_subject_refs=member_refs,
        direct_pair_refs=(pair_refs[0],),
        exact_pair_count=1,
        near_pair_count=0,
        policy_ref=policy_ref,
        audit=_audit(*member_refs, pair_refs[0], policy_ref),
    )
    with pytest.raises(ValidationError, match="direct pairs"):
        DuplicateDetectionResultV2.create(
            resolved_job_work_graph_ref=graph_ref,
            policy_ref=policy_ref,
            item_quality_result_refs=item_quality_refs,
            task_fingerprints=tasks,
            attachment_fingerprints=(),
            duplicate_pairs=pairs,
            duplicate_clusters=(disconnected,),
            audit=_audit(
                graph_ref,
                policy_ref,
                *item_quality_refs,
                *(task_duplicate_fingerprint_v2_ref(item) for item in tasks),
                *pair_refs,
                duplicate_cluster_v2_ref(disconnected),
            ),
        )
