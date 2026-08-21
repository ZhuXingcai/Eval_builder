from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from env_mock_agent.facade import (
    ATTACHMENT_DUPLICATE_NORMALIZATION_VERSION,
    AttachmentDuplicateFingerprintLimitsV2,
    AttachmentDuplicateFingerprintOutcomeV2,
    AttachmentDuplicateFingerprintResultV2,
    attachment_duplicate_fingerprint_result_ref,
    validate_attachment_duplicate_fingerprint_result_identity,
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

DUPLICATE_DETECTION_POLICY_VERSION: Literal["duplicate-detection/r7-01-v1"] = "duplicate-detection/r7-01-v1"
TASK_DUPLICATE_NORMALIZATION_VERSION: Literal["task-duplicate-normalization/r7-01-v1"] = (
    "task-duplicate-normalization/r7-01-v1"
)


class DuplicateSubjectKindV2(StrEnum):
    TASK = "TASK"
    ATTACHMENT = "ATTACHMENT"


class DuplicateMatchKindV2(StrEnum):
    EXACT = "EXACT"
    NEAR = "NEAR"


class DuplicateDetectionPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/duplicate-detection-policy/v2"] = (
        "eval-factory/duplicate-detection-policy/v2"
    )
    duplicate_detection_policy_id: Identifier
    task_near_threshold_bps: int = Field(ge=0, le=10_000)
    attachment_near_threshold_bps: int = Field(ge=0, le=10_000)
    task_shingle_size: int = Field(ge=1, le=16)
    attachment_shingle_size: int = Field(ge=1, le=16)
    fingerprint_bits: Literal[256]
    min_task_tokens: int = Field(ge=1, le=100_000)
    min_attachment_tokens: int = Field(ge=1, le=100_000)
    max_task_characters: int = Field(ge=1, le=10_000_000)
    max_attachment_characters: int = Field(ge=1, le=10_000_000)
    max_items: int = Field(ge=1, le=100_000)
    max_attachments: int = Field(ge=0, le=1_000_000)
    max_pair_comparisons: int = Field(ge=0, le=100_000_000)
    attachment_fingerprint_limits: AttachmentDuplicateFingerprintLimitsV2
    clustering_mode: Literal["CONNECTED_COMPONENTS"] = "CONNECTED_COMPONENTS"
    task_normalization_version: Literal["task-duplicate-normalization/r7-01-v1"] = (
        TASK_DUPLICATE_NORMALIZATION_VERSION
    )
    attachment_normalization_version: Literal["attachment-duplicate-normalization/r7-01-v1"] = (
        ATTACHMENT_DUPLICATE_NORMALIZATION_VERSION
    )
    policy_version: Literal["duplicate-detection/r7-01-v1"] = DUPLICATE_DETECTION_POLICY_VERSION
    policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        limits = self.attachment_fingerprint_limits
        if (
            self.attachment_shingle_size != limits.shingle_size
            or self.fingerprint_bits != limits.fingerprint_bits
            or self.min_attachment_tokens != limits.min_token_count
            or self.max_attachment_characters != limits.max_extracted_characters
        ):
            raise ValueError("attachment duplicate policy must match facade limits")
        _validate_audit(self.audit, (), "duplicate detection policy")
        _validate_identity(
            object_id=self.duplicate_detection_policy_id,
            object_sha256=self.policy_sha256,
            expected_prefix="duplicate-detection-policy",
            observed=duplicate_detection_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        task_near_threshold_bps: int,
        attachment_near_threshold_bps: int,
        task_shingle_size: int,
        attachment_shingle_size: int,
        fingerprint_bits: Literal[256],
        min_task_tokens: int,
        min_attachment_tokens: int,
        max_task_characters: int,
        max_attachment_characters: int,
        max_items: int,
        max_attachments: int,
        max_pair_comparisons: int,
        attachment_fingerprint_limits: (AttachmentDuplicateFingerprintLimitsV2),
        audit: ContractAudit,
    ) -> DuplicateDetectionPolicyV2:
        pending = cls(
            duplicate_detection_policy_id=("duplicate-detection-policy://pending"),
            task_near_threshold_bps=task_near_threshold_bps,
            attachment_near_threshold_bps=(attachment_near_threshold_bps),
            task_shingle_size=task_shingle_size,
            attachment_shingle_size=attachment_shingle_size,
            fingerprint_bits=fingerprint_bits,
            min_task_tokens=min_task_tokens,
            min_attachment_tokens=min_attachment_tokens,
            max_task_characters=max_task_characters,
            max_attachment_characters=max_attachment_characters,
            max_items=max_items,
            max_attachments=max_attachments,
            max_pair_comparisons=max_pair_comparisons,
            attachment_fingerprint_limits=(attachment_fingerprint_limits),
            policy_sha256="0" * 64,
            audit=audit,
        )
        digest = duplicate_detection_policy_v2_carried_sha256(pending)
        return _finalize(
            pending,
            id_field="duplicate_detection_policy_id",
            hash_field="policy_sha256",
            prefix="duplicate-detection-policy",
            digest=digest,
        )


class TaskDuplicateFingerprintV2(ContractModelV2):
    schema_version: Literal["eval-factory/task-duplicate-fingerprint/v2"] = (
        "eval-factory/task-duplicate-fingerprint/v2"
    )
    item_id: Identifier
    task_draft_ref: ObjectRef
    r4_task_contract_set_ref: ObjectRef
    exact_task_sha256: Sha256
    similarity_fingerprint: Sha256
    token_count: int = Field(ge=1, le=10_000_000)
    shingle_count: int = Field(ge=1, le=10_000_000)
    normalization_version: Literal["task-duplicate-normalization/r7-01-v1"] = (
        TASK_DUPLICATE_NORMALIZATION_VERSION
    )
    policy_ref: ObjectRef
    fingerprint_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_fingerprint(self) -> Self:
        _require_ref(
            self.task_draft_ref,
            "task-draft",
            "v2",
            "task_draft_ref",
        )
        _require_ref(
            self.r4_task_contract_set_ref,
            "r4-task-contract-set",
            "v2",
            "r4_task_contract_set_ref",
        )
        _require_ref(
            self.policy_ref,
            "duplicate-detection-policy",
            "v2",
            "policy_ref",
        )
        _validate_audit(
            self.audit,
            (
                self.task_draft_ref,
                self.r4_task_contract_set_ref,
                self.policy_ref,
            ),
            "task duplicate fingerprint",
        )
        _validate_identity(
            object_id=(f"task-duplicate-fingerprint://sha256/{self.fingerprint_sha256}"),
            object_sha256=self.fingerprint_sha256,
            expected_prefix="task-duplicate-fingerprint",
            observed=task_duplicate_fingerprint_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        item_id: str,
        task_draft_ref: ObjectRef,
        r4_task_contract_set_ref: ObjectRef,
        exact_task_sha256: str,
        similarity_fingerprint: str,
        token_count: int,
        shingle_count: int,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> TaskDuplicateFingerprintV2:
        pending = cls(
            item_id=item_id,
            task_draft_ref=task_draft_ref,
            r4_task_contract_set_ref=r4_task_contract_set_ref,
            exact_task_sha256=exact_task_sha256,
            similarity_fingerprint=similarity_fingerprint,
            token_count=token_count,
            shingle_count=shingle_count,
            policy_ref=policy_ref,
            fingerprint_sha256="0" * 64,
            audit=audit,
        )
        digest = task_duplicate_fingerprint_v2_carried_sha256(pending)
        return pending.model_copy(update={"fingerprint_sha256": digest})


class AttachmentDuplicateFingerprintV2(ContractModelV2):
    schema_version: Literal["eval-factory/attachment-duplicate-fingerprint/v2"] = (
        "eval-factory/attachment-duplicate-fingerprint/v2"
    )
    item_id: Identifier
    item_quality_result_ref: ObjectRef
    environment_artifact_ref: ObjectRef
    candidate_artifact_version_ref: ObjectRef
    output_ref: ObjectRef
    artifact_validation_result_ref: ObjectRef
    facade_request_ref: ObjectRef
    facade_result: AttachmentDuplicateFingerprintResultV2
    facade_result_ref: ObjectRef
    policy_ref: ObjectRef
    fingerprint_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_fingerprint(self) -> Self:
        for ref, object_type, field_name in (
            (
                self.item_quality_result_ref,
                "item-quality-compilation-result",
                "item_quality_result_ref",
            ),
            (
                self.environment_artifact_ref,
                "environment-artifact",
                "environment_artifact_ref",
            ),
            (
                self.candidate_artifact_version_ref,
                "candidate-artifact-version",
                "candidate_artifact_version_ref",
            ),
            (self.output_ref, "attachment-output", "output_ref"),
            (
                self.artifact_validation_result_ref,
                "artifact-deterministic-validation-result",
                "artifact_validation_result_ref",
            ),
            (
                self.facade_request_ref,
                "attachment-duplicate-fingerprint-request",
                "facade_request_ref",
            ),
            (
                self.facade_result_ref,
                "attachment-duplicate-fingerprint-result",
                "facade_result_ref",
            ),
            (
                self.policy_ref,
                "duplicate-detection-policy",
                "policy_ref",
            ),
        ):
            _require_ref(ref, object_type, "v2", field_name)
        validate_attachment_duplicate_fingerprint_result_identity(self.facade_result)
        expected_result_ref = attachment_duplicate_fingerprint_result_ref(self.facade_result)
        if not _facade_ref_matches_object(
            expected_result_ref,
            self.facade_result_ref,
        ):
            raise ValueError("facade_result_ref must match facade_result identity")
        if not _facade_ref_matches_object(
            self.facade_result.fingerprint_request_ref,
            self.facade_request_ref,
        ):
            raise ValueError("facade_request_ref must match facade result request")
        if not _facade_ref_matches_object(
            self.facade_result.environment_artifact_ref,
            self.environment_artifact_ref,
        ):
            raise ValueError("facade result environment artifact does not match subject")
        if not _facade_ref_matches_object(
            self.facade_result.output_ref,
            self.output_ref,
        ):
            raise ValueError("facade result output does not match attachment subject")
        _validate_audit(
            self.audit,
            (
                self.item_quality_result_ref,
                self.environment_artifact_ref,
                self.candidate_artifact_version_ref,
                self.output_ref,
                self.artifact_validation_result_ref,
                self.facade_request_ref,
                self.facade_result_ref,
                self.policy_ref,
            ),
            "attachment duplicate fingerprint",
        )
        _validate_identity(
            object_id=(f"attachment-duplicate-fingerprint://sha256/{self.fingerprint_sha256}"),
            object_sha256=self.fingerprint_sha256,
            expected_prefix="attachment-duplicate-fingerprint",
            observed=(attachment_duplicate_fingerprint_v2_carried_sha256(self)),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        item_id: str,
        item_quality_result_ref: ObjectRef,
        environment_artifact_ref: ObjectRef,
        candidate_artifact_version_ref: ObjectRef,
        output_ref: ObjectRef,
        artifact_validation_result_ref: ObjectRef,
        facade_request_ref: ObjectRef,
        facade_result: AttachmentDuplicateFingerprintResultV2,
        facade_result_ref: ObjectRef,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> AttachmentDuplicateFingerprintV2:
        pending = cls(
            item_id=item_id,
            item_quality_result_ref=item_quality_result_ref,
            environment_artifact_ref=environment_artifact_ref,
            candidate_artifact_version_ref=(candidate_artifact_version_ref),
            output_ref=output_ref,
            artifact_validation_result_ref=(artifact_validation_result_ref),
            facade_request_ref=facade_request_ref,
            facade_result=facade_result,
            facade_result_ref=facade_result_ref,
            policy_ref=policy_ref,
            fingerprint_sha256="0" * 64,
            audit=audit,
        )
        digest = attachment_duplicate_fingerprint_v2_carried_sha256(pending)
        return pending.model_copy(update={"fingerprint_sha256": digest})


class DuplicatePairEvidenceV2(ContractModelV2):
    schema_version: Literal["eval-factory/duplicate-pair-evidence/v2"] = (
        "eval-factory/duplicate-pair-evidence/v2"
    )
    duplicate_pair_id: Identifier
    subject_kind: DuplicateSubjectKindV2
    left_item_id: Identifier
    right_item_id: Identifier
    left_subject_ref: ObjectRef
    right_subject_ref: ObjectRef
    left_fingerprint_ref: ObjectRef
    right_fingerprint_ref: ObjectRef
    match_kind: DuplicateMatchKindV2
    similarity_bps: int = Field(ge=0, le=10_000)
    policy_ref: ObjectRef
    pair_sha256: Sha256
    audit: ContractAudit

    @field_validator("subject_kind", mode="before")
    @classmethod
    def parse_subject_kind(
        cls,
        value: object,
    ) -> DuplicateSubjectKindV2:
        return _parse_enum(value, DuplicateSubjectKindV2, "subject_kind")

    @field_validator("match_kind", mode="before")
    @classmethod
    def parse_match_kind(
        cls,
        value: object,
    ) -> DuplicateMatchKindV2:
        return _parse_enum(value, DuplicateMatchKindV2, "match_kind")

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        if self.left_item_id == self.right_item_id:
            raise ValueError("duplicate pair must compare distinct Items")
        left_key = (_ref_key(self.left_subject_ref), self.left_item_id)
        right_key = (_ref_key(self.right_subject_ref), self.right_item_id)
        if left_key >= right_key:
            raise ValueError("duplicate pair subjects must be canonical")
        subject_type, fingerprint_type = _subject_ref_types(self.subject_kind)
        for ref, object_type, field_name in (
            (self.left_subject_ref, subject_type, "left_subject_ref"),
            (self.right_subject_ref, subject_type, "right_subject_ref"),
            (
                self.left_fingerprint_ref,
                fingerprint_type,
                "left_fingerprint_ref",
            ),
            (
                self.right_fingerprint_ref,
                fingerprint_type,
                "right_fingerprint_ref",
            ),
            (
                self.policy_ref,
                "duplicate-detection-policy",
                "policy_ref",
            ),
        ):
            _require_ref(ref, object_type, "v2", field_name)
        if self.match_kind is DuplicateMatchKindV2.EXACT and self.similarity_bps != 10_000:
            raise ValueError("EXACT duplicate pair requires 10000 similarity_bps")
        _validate_audit(
            self.audit,
            (
                self.left_subject_ref,
                self.right_subject_ref,
                self.left_fingerprint_ref,
                self.right_fingerprint_ref,
                self.policy_ref,
            ),
            "duplicate pair",
        )
        _validate_identity(
            object_id=self.duplicate_pair_id,
            object_sha256=self.pair_sha256,
            expected_prefix="duplicate-pair-evidence",
            observed=duplicate_pair_evidence_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        subject_kind: DuplicateSubjectKindV2,
        left_item_id: str,
        right_item_id: str,
        left_subject_ref: ObjectRef,
        right_subject_ref: ObjectRef,
        left_fingerprint_ref: ObjectRef,
        right_fingerprint_ref: ObjectRef,
        match_kind: DuplicateMatchKindV2,
        similarity_bps: int,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> DuplicatePairEvidenceV2:
        left = (
            left_item_id,
            left_subject_ref,
            left_fingerprint_ref,
        )
        right = (
            right_item_id,
            right_subject_ref,
            right_fingerprint_ref,
        )
        if (_ref_key(left_subject_ref), left_item_id) > (
            _ref_key(right_subject_ref),
            right_item_id,
        ):
            left, right = right, left
        pending = cls(
            duplicate_pair_id="duplicate-pair-evidence://pending",
            subject_kind=subject_kind,
            left_item_id=left[0],
            right_item_id=right[0],
            left_subject_ref=left[1],
            right_subject_ref=right[1],
            left_fingerprint_ref=left[2],
            right_fingerprint_ref=right[2],
            match_kind=match_kind,
            similarity_bps=similarity_bps,
            policy_ref=policy_ref,
            pair_sha256="0" * 64,
            audit=audit,
        )
        digest = duplicate_pair_evidence_v2_carried_sha256(pending)
        return _finalize(
            pending,
            id_field="duplicate_pair_id",
            hash_field="pair_sha256",
            prefix="duplicate-pair-evidence",
            digest=digest,
        )


class DuplicateClusterV2(ContractModelV2):
    schema_version: Literal["eval-factory/duplicate-cluster/v2"] = "eval-factory/duplicate-cluster/v2"
    duplicate_cluster_id: Identifier
    subject_kind: DuplicateSubjectKindV2
    member_item_ids: tuple[Identifier, ...] = Field(min_length=2)
    member_subject_refs: tuple[ObjectRef, ...] = Field(min_length=2)
    direct_pair_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    exact_pair_count: int = Field(ge=0)
    near_pair_count: int = Field(ge=0)
    clustering_mode: Literal["CONNECTED_COMPONENTS"] = "CONNECTED_COMPONENTS"
    policy_ref: ObjectRef
    cluster_sha256: Sha256
    audit: ContractAudit

    @field_validator("subject_kind", mode="before")
    @classmethod
    def parse_subject_kind(
        cls,
        value: object,
    ) -> DuplicateSubjectKindV2:
        return _parse_enum(value, DuplicateSubjectKindV2, "subject_kind")

    @model_validator(mode="after")
    def validate_cluster(self) -> Self:
        if len(self.member_item_ids) != len(self.member_subject_refs):
            raise ValueError("cluster item IDs must align with member subject refs")
        member_pairs = tuple(
            zip(
                self.member_subject_refs,
                self.member_item_ids,
                strict=True,
            )
        )
        if member_pairs != tuple(
            sorted(
                member_pairs,
                key=lambda item: (_ref_key(item[0]), item[1]),
            )
        ):
            raise ValueError("cluster members must be canonical")
        _require_sorted_unique_refs(
            "cluster member subject refs",
            self.member_subject_refs,
        )
        _require_sorted_unique_refs(
            "cluster direct pair refs",
            self.direct_pair_refs,
        )
        subject_type, _ = _subject_ref_types(self.subject_kind)
        for ref in self.member_subject_refs:
            _require_ref(
                ref,
                subject_type,
                "v2",
                "member_subject_refs",
            )
        for ref in self.direct_pair_refs:
            _require_ref(
                ref,
                "duplicate-pair-evidence",
                "v2",
                "direct_pair_refs",
            )
        _require_ref(
            self.policy_ref,
            "duplicate-detection-policy",
            "v2",
            "policy_ref",
        )
        if self.exact_pair_count + self.near_pair_count != len(self.direct_pair_refs):
            raise ValueError("cluster pair counts must classify direct pair refs")
        _validate_audit(
            self.audit,
            (
                *self.member_subject_refs,
                *self.direct_pair_refs,
                self.policy_ref,
            ),
            "duplicate cluster",
        )
        _validate_identity(
            object_id=self.duplicate_cluster_id,
            object_sha256=self.cluster_sha256,
            expected_prefix="duplicate-cluster",
            observed=duplicate_cluster_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        subject_kind: DuplicateSubjectKindV2,
        member_item_ids: tuple[str, ...],
        member_subject_refs: tuple[ObjectRef, ...],
        direct_pair_refs: tuple[ObjectRef, ...],
        exact_pair_count: int,
        near_pair_count: int,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> DuplicateClusterV2:
        members = tuple(
            sorted(
                zip(
                    member_subject_refs,
                    member_item_ids,
                    strict=True,
                ),
                key=lambda item: (_ref_key(item[0]), item[1]),
            )
        )
        pending = cls(
            duplicate_cluster_id="duplicate-cluster://pending",
            subject_kind=subject_kind,
            member_item_ids=tuple(item[1] for item in members),
            member_subject_refs=tuple(item[0] for item in members),
            direct_pair_refs=_sorted_refs(direct_pair_refs),
            exact_pair_count=exact_pair_count,
            near_pair_count=near_pair_count,
            policy_ref=policy_ref,
            cluster_sha256="0" * 64,
            audit=audit,
        )
        digest = duplicate_cluster_v2_carried_sha256(pending)
        return _finalize(
            pending,
            id_field="duplicate_cluster_id",
            hash_field="cluster_sha256",
            prefix="duplicate-cluster",
            digest=digest,
        )


class DuplicateDetectionResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/duplicate-detection-result/v2"] = (
        "eval-factory/duplicate-detection-result/v2"
    )
    duplicate_detection_result_id: Identifier
    resolved_job_work_graph_ref: ObjectRef
    policy_ref: ObjectRef
    item_quality_result_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    task_fingerprints: tuple[TaskDuplicateFingerprintV2, ...] = Field(min_length=1)
    attachment_fingerprints: tuple[
        AttachmentDuplicateFingerprintV2,
        ...,
    ] = ()
    unsupported_attachment_subject_refs: tuple[ObjectRef, ...] = ()
    duplicate_pairs: tuple[DuplicatePairEvidenceV2, ...] = ()
    duplicate_clusters: tuple[DuplicateClusterV2, ...] = ()
    evaluated_task_pair_count: int = Field(ge=0)
    evaluated_attachment_pair_count: int = Field(ge=0)
    unsupported_attachment_count: int = Field(ge=0)
    policy_version: Literal["duplicate-detection/r7-01-v1"] = DUPLICATE_DETECTION_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _require_ref(
            self.resolved_job_work_graph_ref,
            "resolved-job-work-graph",
            "v2",
            "resolved_job_work_graph_ref",
        )
        _require_ref(
            self.policy_ref,
            "duplicate-detection-policy",
            "v2",
            "policy_ref",
        )
        _require_sorted_unique_refs(
            "item quality result refs",
            self.item_quality_result_refs,
        )
        for ref in self.item_quality_result_refs:
            _require_ref(
                ref,
                "item-quality-compilation-result",
                "v2",
                "item_quality_result_refs",
            )
        for task_value in self.task_fingerprints:
            validate_task_duplicate_fingerprint_v2_identity(task_value)
            if task_value.policy_ref != self.policy_ref:
                raise ValueError("task fingerprint policy does not match result")
        for attachment_value in self.attachment_fingerprints:
            validate_attachment_duplicate_fingerprint_v2_identity(attachment_value)
            if attachment_value.policy_ref != self.policy_ref:
                raise ValueError("attachment fingerprint policy does not match result")
        for pair_value in self.duplicate_pairs:
            validate_duplicate_pair_evidence_v2_identity(pair_value)
            if pair_value.policy_ref != self.policy_ref:
                raise ValueError("pair policy does not match result")
        for cluster_value in self.duplicate_clusters:
            validate_duplicate_cluster_v2_identity(cluster_value)
            if cluster_value.policy_ref != self.policy_ref:
                raise ValueError("cluster policy does not match result")
        _require_nested_order(
            "task fingerprints",
            tuple(task_duplicate_fingerprint_v2_ref(value) for value in self.task_fingerprints),
        )
        _require_nested_order(
            "attachment fingerprints",
            tuple(attachment_duplicate_fingerprint_v2_ref(value) for value in self.attachment_fingerprints),
        )
        _require_nested_order(
            "duplicate pairs",
            tuple(duplicate_pair_evidence_v2_ref(value) for value in self.duplicate_pairs),
        )
        _require_nested_order(
            "duplicate clusters",
            tuple(duplicate_cluster_v2_ref(value) for value in self.duplicate_clusters),
        )
        task_item_ids = tuple(value.item_id for value in self.task_fingerprints)
        if len(set(task_item_ids)) != len(task_item_ids) or len(task_item_ids) != len(
            self.item_quality_result_refs
        ):
            raise ValueError("task fingerprints must exactly classify requested Items")
        _require_unique_refs(
            "task fingerprint subjects",
            tuple(value.task_draft_ref for value in self.task_fingerprints),
        )
        item_id_set = set(task_item_ids)
        if any(value.item_id not in item_id_set for value in self.attachment_fingerprints):
            raise ValueError("attachment fingerprint references unknown Item")
        if any(
            value.item_quality_result_ref not in self.item_quality_result_refs
            for value in self.attachment_fingerprints
        ):
            raise ValueError("attachment fingerprint references unknown item quality result")
        if any(
            value.facade_result.outcome is AttachmentDuplicateFingerprintOutcomeV2.BLOCKED
            for value in self.attachment_fingerprints
        ):
            raise ValueError("blocked attachment fingerprint cannot enter a result")
        attachment_subject_refs = tuple(
            value.environment_artifact_ref for value in self.attachment_fingerprints
        )
        _require_unique_refs(
            "attachment fingerprint subjects",
            attachment_subject_refs,
        )
        expected_unsupported = _sorted_refs(
            tuple(
                value.environment_artifact_ref
                for value in self.attachment_fingerprints
                if value.facade_result.outcome is AttachmentDuplicateFingerprintOutcomeV2.UNSUPPORTED
            )
        )
        if (
            self.unsupported_attachment_subject_refs != expected_unsupported
            or self.unsupported_attachment_count != len(expected_unsupported)
        ):
            raise ValueError("unsupported attachment inventory is not exact")
        _require_sorted_unique_refs(
            "unsupported attachment subject refs",
            self.unsupported_attachment_subject_refs,
        )
        expected_task_pairs = len(self.task_fingerprints) * (len(self.task_fingerprints) - 1) // 2
        expected_attachment_pairs = sum(
            1
            for left_index, left in enumerate(self.attachment_fingerprints)
            for right in self.attachment_fingerprints[left_index + 1 :]
            if left.item_id != right.item_id
        )
        if (
            self.evaluated_task_pair_count != expected_task_pairs
            or self.evaluated_attachment_pair_count != expected_attachment_pairs
        ):
            raise ValueError("evaluated pair counts do not match source inventory")
        _validate_result_pairs_and_clusters(self)
        _validate_audit(
            self.audit,
            _duplicate_result_refs(self),
            "duplicate detection result",
        )
        _validate_identity(
            object_id=self.duplicate_detection_result_id,
            object_sha256=self.result_sha256,
            expected_prefix="duplicate-detection-result",
            observed=duplicate_detection_result_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        resolved_job_work_graph_ref: ObjectRef,
        policy_ref: ObjectRef,
        item_quality_result_refs: tuple[ObjectRef, ...],
        task_fingerprints: tuple[TaskDuplicateFingerprintV2, ...],
        attachment_fingerprints: tuple[
            AttachmentDuplicateFingerprintV2,
            ...,
        ],
        duplicate_pairs: tuple[DuplicatePairEvidenceV2, ...],
        duplicate_clusters: tuple[DuplicateClusterV2, ...],
        audit: ContractAudit,
    ) -> DuplicateDetectionResultV2:
        sorted_tasks = tuple(
            sorted(
                task_fingerprints,
                key=lambda value: _ref_key(task_duplicate_fingerprint_v2_ref(value)),
            )
        )
        sorted_attachments = tuple(
            sorted(
                attachment_fingerprints,
                key=lambda value: _ref_key(attachment_duplicate_fingerprint_v2_ref(value)),
            )
        )
        sorted_pairs = tuple(
            sorted(
                duplicate_pairs,
                key=lambda value: _ref_key(duplicate_pair_evidence_v2_ref(value)),
            )
        )
        sorted_clusters = tuple(
            sorted(
                duplicate_clusters,
                key=lambda value: _ref_key(duplicate_cluster_v2_ref(value)),
            )
        )
        unsupported = _sorted_refs(
            tuple(
                value.environment_artifact_ref
                for value in sorted_attachments
                if value.facade_result.outcome is AttachmentDuplicateFingerprintOutcomeV2.UNSUPPORTED
            )
        )
        pending = cls(
            duplicate_detection_result_id=("duplicate-detection-result://pending"),
            resolved_job_work_graph_ref=resolved_job_work_graph_ref,
            policy_ref=policy_ref,
            item_quality_result_refs=_sorted_refs(item_quality_result_refs),
            task_fingerprints=sorted_tasks,
            attachment_fingerprints=sorted_attachments,
            unsupported_attachment_subject_refs=unsupported,
            duplicate_pairs=sorted_pairs,
            duplicate_clusters=sorted_clusters,
            evaluated_task_pair_count=len(sorted_tasks) * (len(sorted_tasks) - 1) // 2,
            evaluated_attachment_pair_count=sum(
                1
                for left_index, left in enumerate(sorted_attachments)
                for right in sorted_attachments[left_index + 1 :]
                if left.item_id != right.item_id
            ),
            unsupported_attachment_count=len(unsupported),
            result_sha256="0" * 64,
            audit=audit,
        )
        digest = duplicate_detection_result_v2_carried_sha256(pending)
        return _finalize(
            pending,
            id_field="duplicate_detection_result_id",
            hash_field="result_sha256",
            prefix="duplicate-detection-result",
            digest=digest,
        )


def duplicate_detection_policy_v2_carried_sha256(
    value: DuplicateDetectionPolicyV2,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={
                "duplicate_detection_policy_id",
                "policy_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def duplicate_detection_policy_v2_ref(
    value: DuplicateDetectionPolicyV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="duplicate-detection-policy",
        object_id=value.duplicate_detection_policy_id,
        object_version="v2",
        object_sha256=value.policy_sha256,
    )


def task_duplicate_fingerprint_v2_carried_sha256(
    value: TaskDuplicateFingerprintV2,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={"fingerprint_sha256", "audit"},
            exclude_none=False,
        )
    )


def task_duplicate_fingerprint_v2_ref(
    value: TaskDuplicateFingerprintV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="task-duplicate-fingerprint",
        object_id=(f"task-duplicate-fingerprint://sha256/{value.fingerprint_sha256}"),
        object_version="v2",
        object_sha256=value.fingerprint_sha256,
    )


def attachment_duplicate_fingerprint_v2_carried_sha256(
    value: AttachmentDuplicateFingerprintV2,
) -> str:
    return _payload_sha256(
        {
            "item_id": value.item_id,
            "item_quality_result_ref": _ref_payload(value.item_quality_result_ref),
            "environment_artifact_ref": _ref_payload(value.environment_artifact_ref),
            "candidate_artifact_version_ref": _ref_payload(value.candidate_artifact_version_ref),
            "output_ref": _ref_payload(value.output_ref),
            "artifact_validation_result_ref": _ref_payload(value.artifact_validation_result_ref),
            "facade_request_ref": _ref_payload(value.facade_request_ref),
            "facade_result_ref": _ref_payload(value.facade_result_ref),
            "policy_ref": _ref_payload(value.policy_ref),
        }
    )


def attachment_duplicate_fingerprint_v2_ref(
    value: AttachmentDuplicateFingerprintV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="attachment-duplicate-fingerprint",
        object_id=(f"attachment-duplicate-fingerprint://sha256/{value.fingerprint_sha256}"),
        object_version="v2",
        object_sha256=value.fingerprint_sha256,
    )


def duplicate_pair_evidence_v2_carried_sha256(
    value: DuplicatePairEvidenceV2,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={
                "duplicate_pair_id",
                "pair_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def duplicate_pair_evidence_v2_ref(
    value: DuplicatePairEvidenceV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="duplicate-pair-evidence",
        object_id=value.duplicate_pair_id,
        object_version="v2",
        object_sha256=value.pair_sha256,
    )


def duplicate_cluster_v2_carried_sha256(
    value: DuplicateClusterV2,
) -> str:
    return _payload_sha256(
        value.model_dump(
            mode="json",
            exclude={
                "duplicate_cluster_id",
                "cluster_sha256",
                "audit",
            },
            exclude_none=False,
        )
    )


def duplicate_cluster_v2_ref(
    value: DuplicateClusterV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="duplicate-cluster",
        object_id=value.duplicate_cluster_id,
        object_version="v2",
        object_sha256=value.cluster_sha256,
    )


def duplicate_detection_result_v2_carried_sha256(
    value: DuplicateDetectionResultV2,
) -> str:
    return _payload_sha256(
        {
            "resolved_job_work_graph_ref": _ref_payload(value.resolved_job_work_graph_ref),
            "policy_ref": _ref_payload(value.policy_ref),
            "item_quality_result_refs": [_ref_payload(ref) for ref in value.item_quality_result_refs],
            "task_fingerprint_refs": [
                _ref_payload(task_duplicate_fingerprint_v2_ref(item)) for item in value.task_fingerprints
            ],
            "attachment_fingerprint_refs": [
                _ref_payload(attachment_duplicate_fingerprint_v2_ref(item))
                for item in value.attachment_fingerprints
            ],
            "unsupported_attachment_subject_refs": [
                _ref_payload(ref) for ref in value.unsupported_attachment_subject_refs
            ],
            "duplicate_pair_refs": [
                _ref_payload(duplicate_pair_evidence_v2_ref(item)) for item in value.duplicate_pairs
            ],
            "duplicate_cluster_refs": [
                _ref_payload(duplicate_cluster_v2_ref(item)) for item in value.duplicate_clusters
            ],
            "evaluated_task_pair_count": (value.evaluated_task_pair_count),
            "evaluated_attachment_pair_count": (value.evaluated_attachment_pair_count),
            "unsupported_attachment_count": (value.unsupported_attachment_count),
            "policy_version": value.policy_version,
        }
    )


def duplicate_detection_result_v2_ref(
    value: DuplicateDetectionResultV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="duplicate-detection-result",
        object_id=value.duplicate_detection_result_id,
        object_version="v2",
        object_sha256=value.result_sha256,
    )


def validate_duplicate_detection_policy_v2_identity(
    value: DuplicateDetectionPolicyV2,
) -> None:
    _require_current_identity(
        object_id=value.duplicate_detection_policy_id,
        object_sha256=value.policy_sha256,
        prefix="duplicate-detection-policy",
        digest=duplicate_detection_policy_v2_carried_sha256(value),
    )


def validate_task_duplicate_fingerprint_v2_identity(
    value: TaskDuplicateFingerprintV2,
) -> None:
    digest = task_duplicate_fingerprint_v2_carried_sha256(value)
    if value.fingerprint_sha256 != digest:
        raise ValueError("task duplicate fingerprint identity is stale")


def validate_attachment_duplicate_fingerprint_v2_identity(
    value: AttachmentDuplicateFingerprintV2,
) -> None:
    validate_attachment_duplicate_fingerprint_result_identity(value.facade_result)
    digest = attachment_duplicate_fingerprint_v2_carried_sha256(value)
    if value.fingerprint_sha256 != digest:
        raise ValueError("attachment duplicate fingerprint identity is stale")


def validate_duplicate_pair_evidence_v2_identity(
    value: DuplicatePairEvidenceV2,
) -> None:
    _require_current_identity(
        object_id=value.duplicate_pair_id,
        object_sha256=value.pair_sha256,
        prefix="duplicate-pair-evidence",
        digest=duplicate_pair_evidence_v2_carried_sha256(value),
    )


def validate_duplicate_cluster_v2_identity(
    value: DuplicateClusterV2,
) -> None:
    _require_current_identity(
        object_id=value.duplicate_cluster_id,
        object_sha256=value.cluster_sha256,
        prefix="duplicate-cluster",
        digest=duplicate_cluster_v2_carried_sha256(value),
    )


def validate_duplicate_detection_result_v2_identity(
    value: DuplicateDetectionResultV2,
) -> None:
    for task_item in value.task_fingerprints:
        validate_task_duplicate_fingerprint_v2_identity(task_item)
    for attachment_item in value.attachment_fingerprints:
        validate_attachment_duplicate_fingerprint_v2_identity(attachment_item)
    for pair_item in value.duplicate_pairs:
        validate_duplicate_pair_evidence_v2_identity(pair_item)
    for cluster_item in value.duplicate_clusters:
        validate_duplicate_cluster_v2_identity(cluster_item)
    _require_current_identity(
        object_id=value.duplicate_detection_result_id,
        object_sha256=value.result_sha256,
        prefix="duplicate-detection-result",
        digest=duplicate_detection_result_v2_carried_sha256(value),
    )


def _validate_result_pairs_and_clusters(
    value: DuplicateDetectionResultV2,
) -> None:
    task_subjects = {
        item.task_draft_ref: task_duplicate_fingerprint_v2_ref(item) for item in value.task_fingerprints
    }
    attachment_subjects = {
        item.environment_artifact_ref: (attachment_duplicate_fingerprint_v2_ref(item))
        for item in value.attachment_fingerprints
    }
    pair_by_ref = {duplicate_pair_evidence_v2_ref(item): item for item in value.duplicate_pairs}
    pair_subject_keys: set[tuple[DuplicateSubjectKindV2, ObjectRef, ObjectRef]] = set()
    for pair in value.duplicate_pairs:
        subjects = task_subjects if pair.subject_kind is DuplicateSubjectKindV2.TASK else attachment_subjects
        if (
            subjects.get(pair.left_subject_ref) != pair.left_fingerprint_ref
            or subjects.get(pair.right_subject_ref) != pair.right_fingerprint_ref
        ):
            raise ValueError("duplicate pair does not bind result fingerprints")
        key = (
            pair.subject_kind,
            pair.left_subject_ref,
            pair.right_subject_ref,
        )
        if key in pair_subject_keys:
            raise ValueError("duplicate pair subjects must be unique")
        pair_subject_keys.add(key)

    for subject_kind in DuplicateSubjectKindV2:
        expected_components = _pair_components(
            tuple(pair for pair in value.duplicate_pairs if pair.subject_kind is subject_kind)
        )
        observed_clusters = tuple(
            cluster for cluster in value.duplicate_clusters if cluster.subject_kind is subject_kind
        )
        if len(observed_clusters) != len(expected_components):
            raise ValueError("duplicate clusters must exactly cover pair components")
        observed_by_members = {
            frozenset(cluster.member_subject_refs): cluster for cluster in observed_clusters
        }
        if len(observed_by_members) != len(observed_clusters):
            raise ValueError("duplicate subject appears in multiple clusters")
        for member_set, expected_pair_refs in expected_components:
            cluster = observed_by_members.get(member_set)
            if cluster is None:
                raise ValueError("duplicate cluster membership is not connected")
            subjects = task_subjects if subject_kind is DuplicateSubjectKindV2.TASK else attachment_subjects
            if any(ref not in subjects for ref in cluster.member_subject_refs):
                raise ValueError("duplicate cluster references unknown result subject")
            if cluster.direct_pair_refs != expected_pair_refs:
                raise ValueError("duplicate cluster direct pairs are not exact")
            component_pairs = tuple(pair_by_ref[ref] for ref in expected_pair_refs)
            exact_count = sum(pair.match_kind is DuplicateMatchKindV2.EXACT for pair in component_pairs)
            if (
                exact_count != cluster.exact_pair_count
                or len(component_pairs) - exact_count != cluster.near_pair_count
            ):
                raise ValueError("duplicate cluster pair kind counts are stale")


def _pair_components(
    pairs: tuple[DuplicatePairEvidenceV2, ...],
) -> tuple[tuple[frozenset[ObjectRef], tuple[ObjectRef, ...]], ...]:
    adjacency: dict[ObjectRef, set[ObjectRef]] = {}
    for pair in pairs:
        adjacency.setdefault(pair.left_subject_ref, set()).add(pair.right_subject_ref)
        adjacency.setdefault(pair.right_subject_ref, set()).add(pair.left_subject_ref)
    remaining = set(adjacency)
    components: list[tuple[frozenset[ObjectRef], tuple[ObjectRef, ...]]] = []
    while remaining:
        root = min(remaining, key=_ref_key)
        stack = [root]
        members: set[ObjectRef] = set()
        while stack:
            subject = stack.pop()
            if subject in members:
                continue
            members.add(subject)
            stack.extend(adjacency.get(subject, ()))
        remaining.difference_update(members)
        pair_refs = _sorted_refs(
            tuple(
                duplicate_pair_evidence_v2_ref(pair)
                for pair in pairs
                if pair.left_subject_ref in members and pair.right_subject_ref in members
            )
        )
        components.append((frozenset(members), pair_refs))
    return tuple(
        sorted(
            components,
            key=lambda item: tuple(_ref_key(ref) for ref in sorted(item[0], key=_ref_key)),
        )
    )


def _duplicate_result_refs(
    value: DuplicateDetectionResultV2,
) -> tuple[ObjectRef, ...]:
    return (
        value.resolved_job_work_graph_ref,
        value.policy_ref,
        *value.item_quality_result_refs,
        *(task_duplicate_fingerprint_v2_ref(item) for item in value.task_fingerprints),
        *(attachment_duplicate_fingerprint_v2_ref(item) for item in value.attachment_fingerprints),
        *value.unsupported_attachment_subject_refs,
        *(duplicate_pair_evidence_v2_ref(item) for item in value.duplicate_pairs),
        *(duplicate_cluster_v2_ref(item) for item in value.duplicate_clusters),
    )


def _subject_ref_types(
    kind: DuplicateSubjectKindV2,
) -> tuple[str, str]:
    if kind is DuplicateSubjectKindV2.TASK:
        return "task-draft", "task-duplicate-fingerprint"
    return "environment-artifact", "attachment-duplicate-fingerprint"


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


def _facade_ref_matches_object(
    facade_ref: object,
    object_ref: ObjectRef,
) -> bool:
    return (
        getattr(facade_ref, "object_type", None) == object_ref.object_type
        and getattr(facade_ref, "object_id", None) == object_ref.object_id
        and getattr(facade_ref, "object_version", None) == object_ref.object_version
        and getattr(facade_ref, "object_sha256", None) == object_ref.object_sha256
    )


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
    if len(values) != len(set(values)) or values != tuple(sorted(values, key=_ref_key)):
        raise ValueError(f"{label} must be sorted and unique")


def _require_unique_refs(
    label: str,
    values: tuple[ObjectRef, ...],
) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _require_nested_order(
    label: str,
    refs: tuple[ObjectRef, ...],
) -> None:
    _require_sorted_unique_refs(label, refs)


def _require_ref(
    value: ObjectRef,
    object_type: str,
    object_version: str,
    field_name: str,
) -> None:
    if value.object_type != object_type or value.object_version != object_version:
        raise ValueError(f"{field_name} must reference {object_type} {object_version}")


def _validate_audit(
    audit: ContractAudit,
    expected_refs: tuple[ObjectRef, ...],
    label: str,
) -> None:
    if audit.input_refs != _sorted_refs(expected_refs):
        raise ValueError(f"{label} audit refs are incomplete")


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
    prefix: str,
    digest: str,
) -> None:
    if object_sha256 != digest or object_id != f"{prefix}://sha256/{digest}":
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
