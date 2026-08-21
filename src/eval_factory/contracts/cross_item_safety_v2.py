from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from env_mock_agent.facade import (
    ATTACHMENT_CROSS_ITEM_SAFETY_POLICY_VERSION,
    AttachmentCrossItemSafetyScanLimitsV2,
    AttachmentCrossItemSafetyScanResultV2,
    AttachmentCrossItemSafetyScanStatusV2,
    FacadeObjectRef,
    attachment_cross_item_safety_scan_result_ref,
    validate_attachment_cross_item_safety_scan_result_identity,
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
from eval_factory.contracts.task_v2 import (
    PromptLeakageCategoryV2,
    PromptLeakageMatchKindV2,
)

CROSS_ITEM_SAFETY_POLICY_VERSION: Literal["cross-item-safety/r7-02-v1"] = "cross-item-safety/r7-02-v1"
CROSS_ITEM_SAFETY_NORMALIZATION_VERSION: Literal["task-prompt-leakage-normalization/r4-04-v1"] = (
    "task-prompt-leakage-normalization/r4-04-v1"
)


class CrossItemSafetyEvidenceKindV2(StrEnum):
    LEAKAGE = "LEAKAGE"
    CONTAMINATION = "CONTAMINATION"


class CrossItemSafetySurfaceKindV2(StrEnum):
    PROMPT = "PROMPT"
    ATTACHMENT = "ATTACHMENT"


class AnswerReuseMatchKindV2(StrEnum):
    FULL_SOURCE = "FULL_SOURCE"
    PARTIAL_WINDOWS = "PARTIAL_WINDOWS"


class CrossItemSafetyClusterKindV2(StrEnum):
    LEAKAGE = "LEAKAGE"
    CONTAMINATION = "CONTAMINATION"
    ANSWER_REUSE = "ANSWER_REUSE"


_CONTAMINATION_CATEGORIES = frozenset(
    {
        PromptLeakageCategoryV2.FINAL_ANSWER,
        PromptLeakageCategoryV2.COMPLETED_DELIVERABLE,
    }
)
_LEAKAGE_CATEGORIES = frozenset(PromptLeakageCategoryV2) - (_CONTAMINATION_CATEGORIES)


class CrossItemSafetyPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/cross-item-safety-policy/v2"] = (
        "eval-factory/cross-item-safety-policy/v2"
    )
    cross_item_safety_policy_id: Identifier
    answer_reuse_window_token_count: Literal[8] = 8
    min_shared_answer_windows: int = Field(ge=2, le=100_000)
    min_answer_reuse_coverage_bps: int = Field(ge=1, le=10_000)
    max_items: int = Field(ge=2, le=100_000)
    max_attachments: int = Field(ge=0, le=1_000_000)
    max_total_fingerprints: int = Field(ge=0, le=10_000_000)
    max_foreign_fingerprints_per_target: int = Field(
        ge=0,
        le=10_000_000,
    )
    max_prompt_characters: int = Field(ge=1, le=10_000_000)
    max_prompt_scan_comparisons: int = Field(ge=0, le=1_000_000_000)
    max_answer_source_pairs: int = Field(ge=0, le=100_000_000)
    max_answer_window_comparisons: int = Field(ge=0, le=1_000_000_000)
    max_visible_matches: int = Field(ge=0, le=10_000_000)
    max_reuse_pairs: int = Field(ge=0, le=100_000_000)
    max_clusters: int = Field(ge=0, le=10_000_000)
    attachment_scan_limits: AttachmentCrossItemSafetyScanLimitsV2
    classification_mode: Literal["CATEGORY_EXACT"] = "CATEGORY_EXACT"
    clustering_mode: Literal["CONNECTED_COMPONENTS"] = "CONNECTED_COMPONENTS"
    normalization_version: Literal["task-prompt-leakage-normalization/r4-04-v1"] = (
        CROSS_ITEM_SAFETY_NORMALIZATION_VERSION
    )
    attachment_policy_version: Literal["attachment-cross-item-safety/r7-02-v1"] = (
        ATTACHMENT_CROSS_ITEM_SAFETY_POLICY_VERSION
    )
    policy_version: Literal["cross-item-safety/r7-02-v1"] = CROSS_ITEM_SAFETY_POLICY_VERSION
    policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        if (
            self.max_foreign_fingerprints_per_target > self.max_total_fingerprints
            or self.attachment_scan_limits.max_fingerprint_count != self.max_foreign_fingerprints_per_target
        ):
            raise ValueError("cross-item fingerprint limits must agree with facade limits")
        _validate_audit(self.audit, (), "cross-item safety policy")
        _validate_identity(
            object_id=self.cross_item_safety_policy_id,
            object_sha256=self.policy_sha256,
            expected_prefix="cross-item-safety-policy",
            observed=cross_item_safety_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        min_shared_answer_windows: int,
        min_answer_reuse_coverage_bps: int,
        max_items: int,
        max_attachments: int,
        max_total_fingerprints: int,
        max_foreign_fingerprints_per_target: int,
        max_prompt_characters: int,
        max_prompt_scan_comparisons: int,
        max_answer_source_pairs: int,
        max_answer_window_comparisons: int,
        max_visible_matches: int,
        max_reuse_pairs: int,
        max_clusters: int,
        attachment_scan_limits: AttachmentCrossItemSafetyScanLimitsV2,
        audit: ContractAudit,
    ) -> CrossItemSafetyPolicyV2:
        value = cls(
            cross_item_safety_policy_id="cross-item-safety-policy://pending",
            min_shared_answer_windows=min_shared_answer_windows,
            min_answer_reuse_coverage_bps=min_answer_reuse_coverage_bps,
            max_items=max_items,
            max_attachments=max_attachments,
            max_total_fingerprints=max_total_fingerprints,
            max_foreign_fingerprints_per_target=(max_foreign_fingerprints_per_target),
            max_prompt_characters=max_prompt_characters,
            max_prompt_scan_comparisons=max_prompt_scan_comparisons,
            max_answer_source_pairs=max_answer_source_pairs,
            max_answer_window_comparisons=max_answer_window_comparisons,
            max_visible_matches=max_visible_matches,
            max_reuse_pairs=max_reuse_pairs,
            max_clusters=max_clusters,
            attachment_scan_limits=attachment_scan_limits,
            policy_sha256="0" * 64,
            audit=audit,
        )
        digest = cross_item_safety_policy_v2_carried_sha256(value)
        return _finalize(
            value,
            id_field="cross_item_safety_policy_id",
            hash_field="policy_sha256",
            prefix="cross-item-safety-policy",
            digest=digest,
        )


class AttachmentCrossItemSafetyScanEvidenceV2(ContractModelV2):
    schema_version: Literal["eval-factory/attachment-cross-item-safety-scan-evidence/v2"] = (
        "eval-factory/attachment-cross-item-safety-scan-evidence/v2"
    )
    target_item_id: Identifier
    item_quality_result_ref: ObjectRef
    environment_artifact_ref: ObjectRef
    candidate_artifact_version_ref: ObjectRef
    output_ref: ObjectRef
    artifact_validation_result_ref: ObjectRef
    facade_request_ref: ObjectRef
    facade_result: AttachmentCrossItemSafetyScanResultV2
    facade_result_ref: ObjectRef
    foreign_reference_set_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    policy_ref: ObjectRef
    evidence_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
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
                "attachment-cross-item-safety-scan-request",
                "facade_request_ref",
            ),
            (
                self.facade_result_ref,
                "attachment-cross-item-safety-scan-result",
                "facade_result_ref",
            ),
            (
                self.policy_ref,
                "cross-item-safety-policy",
                "policy_ref",
            ),
        ):
            _require_ref(ref, object_type, "v2", field_name)
        _require_sorted_unique_refs(
            "foreign reference sets",
            self.foreign_reference_set_refs,
        )
        for ref in self.foreign_reference_set_refs:
            _require_ref(
                ref,
                "prompt-leakage-reference-set",
                "v2",
                "foreign_reference_set_refs",
            )
        validate_attachment_cross_item_safety_scan_result_identity(self.facade_result)
        if (
            self.facade_result.status is not AttachmentCrossItemSafetyScanStatusV2.PASSED
            or _object_ref_from_facade(self.facade_result.scan_request_ref) != self.facade_request_ref
            or _object_ref_from_facade(self.facade_result.environment_artifact_ref)
            != self.environment_artifact_ref
            or _object_ref_from_facade(self.facade_result.output_ref) != self.output_ref
            or _object_ref_from_facade(attachment_cross_item_safety_scan_result_ref(self.facade_result))
            != self.facade_result_ref
        ):
            raise ValueError("attachment cross-item scan evidence requires a matching passed result")
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
                *self.foreign_reference_set_refs,
                self.policy_ref,
            ),
            "attachment cross-item scan evidence",
        )
        _validate_identity(
            object_id=(f"attachment-cross-item-safety-scan-evidence://sha256/{self.evidence_sha256}"),
            object_sha256=self.evidence_sha256,
            expected_prefix="attachment-cross-item-safety-scan-evidence",
            observed=(attachment_cross_item_safety_scan_evidence_v2_carried_sha256(self)),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        target_item_id: str,
        item_quality_result_ref: FacadeObjectRef,
        environment_artifact_ref: FacadeObjectRef,
        candidate_artifact_version_ref: FacadeObjectRef,
        output_ref: FacadeObjectRef,
        artifact_validation_result_ref: FacadeObjectRef,
        facade_request_ref: FacadeObjectRef,
        facade_result: AttachmentCrossItemSafetyScanResultV2,
        facade_result_ref: FacadeObjectRef,
        foreign_reference_set_refs: tuple[FacadeObjectRef, ...],
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> AttachmentCrossItemSafetyScanEvidenceV2:
        value = cls(
            target_item_id=target_item_id,
            item_quality_result_ref=_object_ref_from_facade(item_quality_result_ref),
            environment_artifact_ref=_object_ref_from_facade(environment_artifact_ref),
            candidate_artifact_version_ref=_object_ref_from_facade(candidate_artifact_version_ref),
            output_ref=_object_ref_from_facade(output_ref),
            artifact_validation_result_ref=_object_ref_from_facade(artifact_validation_result_ref),
            facade_request_ref=_object_ref_from_facade(facade_request_ref),
            facade_result=facade_result,
            facade_result_ref=_object_ref_from_facade(facade_result_ref),
            foreign_reference_set_refs=_sorted_refs(
                tuple(_object_ref_from_facade(ref) for ref in foreign_reference_set_refs)
            ),
            policy_ref=policy_ref,
            evidence_sha256="0" * 64,
            audit=audit,
        )
        digest = attachment_cross_item_safety_scan_evidence_v2_carried_sha256(value)
        return value.model_copy(update={"evidence_sha256": digest})


class CrossItemVisibleMatchEvidenceV2(ContractModelV2):
    schema_version: Literal["eval-factory/cross-item-visible-match-evidence/v2"] = (
        "eval-factory/cross-item-visible-match-evidence/v2"
    )
    visible_match_evidence_id: Identifier
    evidence_kind: CrossItemSafetyEvidenceKindV2
    target_surface: CrossItemSafetySurfaceKindV2
    source_item_id: Identifier
    target_item_id: Identifier
    source_reference_set_ref: ObjectRef
    source_subject_ref: ObjectRef
    target_subject_ref: ObjectRef
    category: PromptLeakageCategoryV2
    matched_fingerprint_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    match_kinds: tuple[PromptLeakageMatchKindV2, ...] = Field(min_length=1)
    attachment_scan_evidence_ref: ObjectRef | None = None
    non_waivable: Literal[True] = True
    policy_ref: ObjectRef
    evidence_sha256: Sha256
    audit: ContractAudit

    @field_validator("evidence_kind", mode="before")
    @classmethod
    def parse_evidence_kind(
        cls,
        value: object,
    ) -> CrossItemSafetyEvidenceKindV2:
        return _parse_enum(
            value,
            CrossItemSafetyEvidenceKindV2,
            "evidence_kind",
        )

    @field_validator("target_surface", mode="before")
    @classmethod
    def parse_target_surface(
        cls,
        value: object,
    ) -> CrossItemSafetySurfaceKindV2:
        return _parse_enum(
            value,
            CrossItemSafetySurfaceKindV2,
            "target_surface",
        )

    @field_validator("category", mode="before")
    @classmethod
    def parse_category(
        cls,
        value: object,
    ) -> PromptLeakageCategoryV2:
        return _parse_enum(value, PromptLeakageCategoryV2, "category")

    @field_validator("match_kinds", mode="before")
    @classmethod
    def parse_match_kinds(
        cls,
        value: object,
    ) -> tuple[PromptLeakageMatchKindV2, ...]:
        if not isinstance(value, (tuple, list, set, frozenset)):
            raise TypeError("match_kinds must be a collection")
        return tuple(
            item if isinstance(item, PromptLeakageMatchKindV2) else PromptLeakageMatchKindV2(item)
            for item in value
        )

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if self.source_item_id == self.target_item_id:
            raise ValueError("cross-item visible evidence requires distinct Items")
        _require_ref(
            self.source_reference_set_ref,
            "prompt-leakage-reference-set",
            "v2",
            "source_reference_set_ref",
        )
        expected_target_type = (
            "task-draft"
            if self.target_surface is CrossItemSafetySurfaceKindV2.PROMPT
            else "environment-artifact"
        )
        _require_ref(
            self.target_subject_ref,
            expected_target_type,
            "v2",
            "target_subject_ref",
        )
        _require_ref(
            self.policy_ref,
            "cross-item-safety-policy",
            "v2",
            "policy_ref",
        )
        _require_sorted_unique_refs(
            "matched fingerprint refs",
            self.matched_fingerprint_refs,
        )
        for ref in self.matched_fingerprint_refs:
            _require_ref(
                ref,
                "prompt-leakage-fingerprint",
                "v2",
                "matched_fingerprint_refs",
            )
        _require_sorted_unique(
            "visible match kinds",
            tuple(value.value for value in self.match_kinds),
        )
        expected_kind = _evidence_kind_for_category(self.category)
        if self.evidence_kind is not expected_kind:
            raise ValueError("visible evidence kind does not match restricted category")
        if self.target_surface is CrossItemSafetySurfaceKindV2.ATTACHMENT:
            if self.attachment_scan_evidence_ref is None:
                raise ValueError("attachment visible evidence requires scan evidence")
            _require_ref(
                self.attachment_scan_evidence_ref,
                "attachment-cross-item-safety-scan-evidence",
                "v2",
                "attachment_scan_evidence_ref",
            )
        elif self.attachment_scan_evidence_ref is not None:
            raise ValueError("prompt visible evidence cannot carry attachment scan")
        refs = [
            self.source_reference_set_ref,
            self.source_subject_ref,
            self.target_subject_ref,
            *self.matched_fingerprint_refs,
            self.policy_ref,
        ]
        if self.attachment_scan_evidence_ref is not None:
            refs.append(self.attachment_scan_evidence_ref)
        _validate_audit(
            self.audit,
            tuple(refs),
            "cross-item visible match evidence",
        )
        _validate_identity(
            object_id=self.visible_match_evidence_id,
            object_sha256=self.evidence_sha256,
            expected_prefix="cross-item-visible-match-evidence",
            observed=cross_item_visible_match_evidence_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        evidence_kind: CrossItemSafetyEvidenceKindV2,
        target_surface: CrossItemSafetySurfaceKindV2,
        source_item_id: str,
        target_item_id: str,
        source_reference_set_ref: ObjectRef,
        source_subject_ref: ObjectRef,
        target_subject_ref: ObjectRef,
        category: PromptLeakageCategoryV2,
        matched_fingerprint_refs: tuple[ObjectRef, ...],
        match_kinds: tuple[PromptLeakageMatchKindV2, ...],
        attachment_scan_evidence_ref: ObjectRef | None,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> CrossItemVisibleMatchEvidenceV2:
        value = cls(
            visible_match_evidence_id=("cross-item-visible-match-evidence://pending"),
            evidence_kind=evidence_kind,
            target_surface=target_surface,
            source_item_id=source_item_id,
            target_item_id=target_item_id,
            source_reference_set_ref=source_reference_set_ref,
            source_subject_ref=source_subject_ref,
            target_subject_ref=target_subject_ref,
            category=category,
            matched_fingerprint_refs=_sorted_refs(matched_fingerprint_refs),
            match_kinds=tuple(
                sorted(
                    set(match_kinds),
                    key=lambda item: item.value,
                )
            ),
            attachment_scan_evidence_ref=attachment_scan_evidence_ref,
            policy_ref=policy_ref,
            evidence_sha256="0" * 64,
            audit=audit,
        )
        digest = cross_item_visible_match_evidence_v2_carried_sha256(value)
        return _finalize(
            value,
            id_field="visible_match_evidence_id",
            hash_field="evidence_sha256",
            prefix="cross-item-visible-match-evidence",
            digest=digest,
        )


class AnswerReusePairEvidenceV2(ContractModelV2):
    schema_version: Literal["eval-factory/answer-reuse-pair-evidence/v2"] = (
        "eval-factory/answer-reuse-pair-evidence/v2"
    )
    answer_reuse_pair_id: Identifier
    left_item_id: Identifier
    right_item_id: Identifier
    left_reference_set_ref: ObjectRef
    right_reference_set_ref: ObjectRef
    left_source_subject_ref: ObjectRef
    right_source_subject_ref: ObjectRef
    left_matched_fingerprint_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    right_matched_fingerprint_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    match_kind: AnswerReuseMatchKindV2
    shared_window_count: int = Field(ge=0)
    shorter_source_window_count: int = Field(ge=0)
    coverage_bps: int = Field(ge=0, le=10_000)
    non_waivable: Literal[True] = True
    policy_ref: ObjectRef
    pair_sha256: Sha256
    audit: ContractAudit

    @field_validator("match_kind", mode="before")
    @classmethod
    def parse_match_kind(
        cls,
        value: object,
    ) -> AnswerReuseMatchKindV2:
        return _parse_enum(value, AnswerReuseMatchKindV2, "match_kind")

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        if self.left_item_id == self.right_item_id:
            raise ValueError("answer reuse pair requires distinct Items")
        left_key = (
            self.left_item_id,
            _ref_key(self.left_source_subject_ref),
        )
        right_key = (
            self.right_item_id,
            _ref_key(self.right_source_subject_ref),
        )
        if left_key >= right_key:
            raise ValueError("answer reuse pair endpoints must be canonical")
        for ref, object_type, field_name in (
            (
                self.left_reference_set_ref,
                "prompt-leakage-reference-set",
                "left_reference_set_ref",
            ),
            (
                self.right_reference_set_ref,
                "prompt-leakage-reference-set",
                "right_reference_set_ref",
            ),
            (
                self.policy_ref,
                "cross-item-safety-policy",
                "policy_ref",
            ),
        ):
            _require_ref(ref, object_type, "v2", field_name)
        for label, refs in (
            (
                "left matched fingerprint refs",
                self.left_matched_fingerprint_refs,
            ),
            (
                "right matched fingerprint refs",
                self.right_matched_fingerprint_refs,
            ),
        ):
            _require_sorted_unique_refs(label, refs)
            for ref in refs:
                _require_ref(
                    ref,
                    "prompt-leakage-fingerprint",
                    "v2",
                    label,
                )
        if self.match_kind is AnswerReuseMatchKindV2.FULL_SOURCE:
            if (
                len(self.left_matched_fingerprint_refs) != 1
                or len(self.right_matched_fingerprint_refs) != 1
                or self.shared_window_count != 0
                or self.shorter_source_window_count != 0
                or self.coverage_bps != 10_000
            ):
                raise ValueError(
                    "FULL_SOURCE answer reuse requires one fingerprint per side and full coverage"
                )
        elif (
            self.shared_window_count < 2
            or self.shorter_source_window_count < self.shared_window_count
            or len(self.left_matched_fingerprint_refs) != self.shared_window_count
            or len(self.right_matched_fingerprint_refs) != self.shared_window_count
            or self.coverage_bps != (self.shared_window_count * 10_000 // self.shorter_source_window_count)
        ):
            raise ValueError("PARTIAL_WINDOWS answer reuse requires exact shared-window coverage")
        _validate_audit(
            self.audit,
            (
                self.left_reference_set_ref,
                self.right_reference_set_ref,
                self.left_source_subject_ref,
                self.right_source_subject_ref,
                *self.left_matched_fingerprint_refs,
                *self.right_matched_fingerprint_refs,
                self.policy_ref,
            ),
            "answer reuse pair",
        )
        _validate_identity(
            object_id=self.answer_reuse_pair_id,
            object_sha256=self.pair_sha256,
            expected_prefix="answer-reuse-pair-evidence",
            observed=answer_reuse_pair_evidence_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        left_item_id: str,
        right_item_id: str,
        left_reference_set_ref: ObjectRef,
        right_reference_set_ref: ObjectRef,
        left_source_subject_ref: ObjectRef,
        right_source_subject_ref: ObjectRef,
        left_matched_fingerprint_refs: tuple[ObjectRef, ...],
        right_matched_fingerprint_refs: tuple[ObjectRef, ...],
        match_kind: AnswerReuseMatchKindV2,
        shared_window_count: int,
        shorter_source_window_count: int,
        coverage_bps: int,
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> AnswerReusePairEvidenceV2:
        left = (
            left_item_id,
            left_reference_set_ref,
            left_source_subject_ref,
            _sorted_refs(left_matched_fingerprint_refs),
        )
        right = (
            right_item_id,
            right_reference_set_ref,
            right_source_subject_ref,
            _sorted_refs(right_matched_fingerprint_refs),
        )
        if (left[0], _ref_key(left[2])) > (
            right[0],
            _ref_key(right[2]),
        ):
            left, right = right, left
        value = cls(
            answer_reuse_pair_id="answer-reuse-pair-evidence://pending",
            left_item_id=left[0],
            right_item_id=right[0],
            left_reference_set_ref=left[1],
            right_reference_set_ref=right[1],
            left_source_subject_ref=left[2],
            right_source_subject_ref=right[2],
            left_matched_fingerprint_refs=left[3],
            right_matched_fingerprint_refs=right[3],
            match_kind=match_kind,
            shared_window_count=shared_window_count,
            shorter_source_window_count=shorter_source_window_count,
            coverage_bps=coverage_bps,
            policy_ref=policy_ref,
            pair_sha256="0" * 64,
            audit=audit,
        )
        digest = answer_reuse_pair_evidence_v2_carried_sha256(value)
        return _finalize(
            value,
            id_field="answer_reuse_pair_id",
            hash_field="pair_sha256",
            prefix="answer-reuse-pair-evidence",
            digest=digest,
        )


class CrossItemSafetyClusterV2(ContractModelV2):
    schema_version: Literal["eval-factory/cross-item-safety-cluster/v2"] = (
        "eval-factory/cross-item-safety-cluster/v2"
    )
    cross_item_safety_cluster_id: Identifier
    cluster_kind: CrossItemSafetyClusterKindV2
    member_item_ids: tuple[Identifier, ...] = Field(min_length=2)
    direct_evidence_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    clustering_mode: Literal["CONNECTED_COMPONENTS"] = "CONNECTED_COMPONENTS"
    policy_ref: ObjectRef
    cluster_sha256: Sha256
    audit: ContractAudit

    @field_validator("cluster_kind", mode="before")
    @classmethod
    def parse_cluster_kind(
        cls,
        value: object,
    ) -> CrossItemSafetyClusterKindV2:
        return _parse_enum(
            value,
            CrossItemSafetyClusterKindV2,
            "cluster_kind",
        )

    @model_validator(mode="after")
    def validate_cluster(self) -> Self:
        _require_sorted_unique("cluster member Item IDs", self.member_item_ids)
        _require_sorted_unique_refs(
            "cluster direct evidence refs",
            self.direct_evidence_refs,
        )
        expected_type = (
            "answer-reuse-pair-evidence"
            if self.cluster_kind is CrossItemSafetyClusterKindV2.ANSWER_REUSE
            else "cross-item-visible-match-evidence"
        )
        for ref in self.direct_evidence_refs:
            _require_ref(
                ref,
                expected_type,
                "v2",
                "direct_evidence_refs",
            )
        _require_ref(
            self.policy_ref,
            "cross-item-safety-policy",
            "v2",
            "policy_ref",
        )
        _validate_audit(
            self.audit,
            (*self.direct_evidence_refs, self.policy_ref),
            "cross-item safety cluster",
        )
        _validate_identity(
            object_id=self.cross_item_safety_cluster_id,
            object_sha256=self.cluster_sha256,
            expected_prefix="cross-item-safety-cluster",
            observed=cross_item_safety_cluster_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        cluster_kind: CrossItemSafetyClusterKindV2,
        member_item_ids: tuple[str, ...],
        direct_evidence_refs: tuple[ObjectRef, ...],
        policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> CrossItemSafetyClusterV2:
        value = cls(
            cross_item_safety_cluster_id="cross-item-safety-cluster://pending",
            cluster_kind=cluster_kind,
            member_item_ids=tuple(sorted(set(member_item_ids))),
            direct_evidence_refs=_sorted_refs(direct_evidence_refs),
            policy_ref=policy_ref,
            cluster_sha256="0" * 64,
            audit=audit,
        )
        digest = cross_item_safety_cluster_v2_carried_sha256(value)
        return _finalize(
            value,
            id_field="cross_item_safety_cluster_id",
            hash_field="cluster_sha256",
            prefix="cross-item-safety-cluster",
            digest=digest,
        )


class CrossItemSafetyResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/cross-item-safety-result/v2"] = (
        "eval-factory/cross-item-safety-result/v2"
    )
    cross_item_safety_result_id: Identifier
    resolved_job_work_graph_ref: ObjectRef
    policy_ref: ObjectRef
    item_ids: tuple[Identifier, ...] = Field(min_length=2)
    task_draft_refs: tuple[ObjectRef, ...] = Field(min_length=2)
    leakage_reference_set_refs: tuple[ObjectRef, ...] = Field(min_length=2)
    item_quality_result_refs: tuple[ObjectRef, ...] = Field(min_length=2)
    answer_source_item_ids: tuple[Identifier, ...] = ()
    answer_source_subject_refs: tuple[ObjectRef, ...] = ()
    attachment_scan_evidence: tuple[
        AttachmentCrossItemSafetyScanEvidenceV2,
        ...,
    ] = ()
    visible_matches: tuple[CrossItemVisibleMatchEvidenceV2, ...] = ()
    answer_reuse_pairs: tuple[AnswerReusePairEvidenceV2, ...] = ()
    clusters: tuple[CrossItemSafetyClusterV2, ...] = ()
    evaluated_prompt_foreign_set_count: int = Field(ge=0)
    evaluated_attachment_foreign_set_count: int = Field(ge=0)
    evaluated_answer_source_pair_count: int = Field(ge=0)
    attachment_scan_count: int = Field(ge=0)
    visible_match_count: int = Field(ge=0)
    answer_reuse_pair_count: int = Field(ge=0)
    cluster_count: int = Field(ge=0)
    policy_version: Literal["cross-item-safety/r7-02-v1"] = CROSS_ITEM_SAFETY_POLICY_VERSION
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
            "cross-item-safety-policy",
            "v2",
            "policy_ref",
        )
        _validate_item_bindings(self)
        for scan_value in self.attachment_scan_evidence:
            validate_attachment_cross_item_safety_scan_evidence_v2_identity(scan_value)
            if scan_value.policy_ref != self.policy_ref:
                raise ValueError("attachment scan policy does not match result")
        for visible_value in self.visible_matches:
            validate_cross_item_visible_match_evidence_v2_identity(visible_value)
            if visible_value.policy_ref != self.policy_ref:
                raise ValueError("visible match policy does not match result")
        for reuse_value in self.answer_reuse_pairs:
            validate_answer_reuse_pair_evidence_v2_identity(reuse_value)
            if reuse_value.policy_ref != self.policy_ref:
                raise ValueError("answer reuse policy does not match result")
        for cluster_value in self.clusters:
            validate_cross_item_safety_cluster_v2_identity(cluster_value)
            if cluster_value.policy_ref != self.policy_ref:
                raise ValueError("cluster policy does not match result")
        _require_nested_order(
            "attachment scan evidence",
            tuple(
                attachment_cross_item_safety_scan_evidence_v2_ref(value)
                for value in self.attachment_scan_evidence
            ),
        )
        _require_nested_order(
            "visible matches",
            tuple(cross_item_visible_match_evidence_v2_ref(value) for value in self.visible_matches),
        )
        _require_nested_order(
            "answer reuse pairs",
            tuple(answer_reuse_pair_evidence_v2_ref(value) for value in self.answer_reuse_pairs),
        )
        _require_nested_order(
            "cross-item safety clusters",
            tuple(cross_item_safety_cluster_v2_ref(value) for value in self.clusters),
        )
        _validate_result_ownership(self)
        _validate_result_counts(self)
        _validate_result_clusters(self)
        _validate_audit(
            self.audit,
            _cross_item_safety_result_refs(self),
            "cross-item safety result",
        )
        _validate_identity(
            object_id=self.cross_item_safety_result_id,
            object_sha256=self.result_sha256,
            expected_prefix="cross-item-safety-result",
            observed=cross_item_safety_result_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        resolved_job_work_graph_ref: ObjectRef,
        policy_ref: ObjectRef,
        item_ids: tuple[str, ...],
        task_draft_refs: tuple[ObjectRef, ...],
        leakage_reference_set_refs: tuple[ObjectRef, ...],
        item_quality_result_refs: tuple[ObjectRef, ...],
        answer_source_item_ids: tuple[str, ...],
        answer_source_subject_refs: tuple[ObjectRef, ...],
        attachment_scan_evidence: tuple[
            AttachmentCrossItemSafetyScanEvidenceV2,
            ...,
        ],
        visible_matches: tuple[CrossItemVisibleMatchEvidenceV2, ...],
        answer_reuse_pairs: tuple[AnswerReusePairEvidenceV2, ...],
        clusters: tuple[CrossItemSafetyClusterV2, ...],
        evaluated_prompt_foreign_set_count: int,
        evaluated_attachment_foreign_set_count: int,
        evaluated_answer_source_pair_count: int,
        audit: ContractAudit,
    ) -> CrossItemSafetyResultV2:
        value = cls(
            cross_item_safety_result_id="cross-item-safety-result://pending",
            resolved_job_work_graph_ref=resolved_job_work_graph_ref,
            policy_ref=policy_ref,
            item_ids=item_ids,
            task_draft_refs=task_draft_refs,
            leakage_reference_set_refs=leakage_reference_set_refs,
            item_quality_result_refs=item_quality_result_refs,
            answer_source_item_ids=answer_source_item_ids,
            answer_source_subject_refs=answer_source_subject_refs,
            attachment_scan_evidence=_sorted_nested(
                attachment_scan_evidence,
                attachment_cross_item_safety_scan_evidence_v2_ref,
            ),
            visible_matches=_sorted_nested(
                visible_matches,
                cross_item_visible_match_evidence_v2_ref,
            ),
            answer_reuse_pairs=_sorted_nested(
                answer_reuse_pairs,
                answer_reuse_pair_evidence_v2_ref,
            ),
            clusters=_sorted_nested(
                clusters,
                cross_item_safety_cluster_v2_ref,
            ),
            evaluated_prompt_foreign_set_count=(evaluated_prompt_foreign_set_count),
            evaluated_attachment_foreign_set_count=(evaluated_attachment_foreign_set_count),
            evaluated_answer_source_pair_count=(evaluated_answer_source_pair_count),
            attachment_scan_count=len(attachment_scan_evidence),
            visible_match_count=len(visible_matches),
            answer_reuse_pair_count=len(answer_reuse_pairs),
            cluster_count=len(clusters),
            result_sha256="0" * 64,
            audit=audit,
        )
        digest = cross_item_safety_result_v2_carried_sha256(value)
        return _finalize(
            value,
            id_field="cross_item_safety_result_id",
            hash_field="result_sha256",
            prefix="cross-item-safety-result",
            digest=digest,
        )


def cross_item_safety_policy_v2_carried_sha256(
    policy: CrossItemSafetyPolicyV2,
) -> str:
    return _carried_sha256(
        policy,
        exclude={
            "cross_item_safety_policy_id",
            "policy_sha256",
            "audit",
        },
    )


def cross_item_safety_policy_v2_ref(
    policy: CrossItemSafetyPolicyV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="cross-item-safety-policy",
        object_id=policy.cross_item_safety_policy_id,
        object_version="v2",
        object_sha256=policy.policy_sha256,
    )


def attachment_cross_item_safety_scan_evidence_v2_carried_sha256(
    evidence: AttachmentCrossItemSafetyScanEvidenceV2,
) -> str:
    return _carried_sha256(
        evidence,
        exclude={"evidence_sha256", "audit"},
    )


def attachment_cross_item_safety_scan_evidence_v2_ref(
    evidence: AttachmentCrossItemSafetyScanEvidenceV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="attachment-cross-item-safety-scan-evidence",
        object_id=(f"attachment-cross-item-safety-scan-evidence://sha256/{evidence.evidence_sha256}"),
        object_version="v2",
        object_sha256=evidence.evidence_sha256,
    )


def cross_item_visible_match_evidence_v2_carried_sha256(
    evidence: CrossItemVisibleMatchEvidenceV2,
) -> str:
    return _carried_sha256(
        evidence,
        exclude={
            "visible_match_evidence_id",
            "evidence_sha256",
            "audit",
        },
    )


def cross_item_visible_match_evidence_v2_ref(
    evidence: CrossItemVisibleMatchEvidenceV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="cross-item-visible-match-evidence",
        object_id=evidence.visible_match_evidence_id,
        object_version="v2",
        object_sha256=evidence.evidence_sha256,
    )


def answer_reuse_pair_evidence_v2_carried_sha256(
    pair: AnswerReusePairEvidenceV2,
) -> str:
    return _carried_sha256(
        pair,
        exclude={"answer_reuse_pair_id", "pair_sha256", "audit"},
    )


def answer_reuse_pair_evidence_v2_ref(
    pair: AnswerReusePairEvidenceV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="answer-reuse-pair-evidence",
        object_id=pair.answer_reuse_pair_id,
        object_version="v2",
        object_sha256=pair.pair_sha256,
    )


def cross_item_safety_cluster_v2_carried_sha256(
    cluster: CrossItemSafetyClusterV2,
) -> str:
    return _carried_sha256(
        cluster,
        exclude={
            "cross_item_safety_cluster_id",
            "cluster_sha256",
            "audit",
        },
    )


def cross_item_safety_cluster_v2_ref(
    cluster: CrossItemSafetyClusterV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="cross-item-safety-cluster",
        object_id=cluster.cross_item_safety_cluster_id,
        object_version="v2",
        object_sha256=cluster.cluster_sha256,
    )


def cross_item_safety_result_v2_carried_sha256(
    result: CrossItemSafetyResultV2,
) -> str:
    return _payload_sha256(
        {
            "resolved_job_work_graph_ref": _ref_payload(result.resolved_job_work_graph_ref),
            "policy_ref": _ref_payload(result.policy_ref),
            "item_ids": list(result.item_ids),
            "task_draft_refs": [_ref_payload(ref) for ref in result.task_draft_refs],
            "leakage_reference_set_refs": [_ref_payload(ref) for ref in result.leakage_reference_set_refs],
            "item_quality_result_refs": [_ref_payload(ref) for ref in result.item_quality_result_refs],
            "answer_source_item_ids": list(result.answer_source_item_ids),
            "answer_source_subject_refs": [_ref_payload(ref) for ref in result.answer_source_subject_refs],
            "attachment_scan_evidence_refs": [
                _ref_payload(attachment_cross_item_safety_scan_evidence_v2_ref(value))
                for value in result.attachment_scan_evidence
            ],
            "visible_match_refs": [
                _ref_payload(cross_item_visible_match_evidence_v2_ref(value))
                for value in result.visible_matches
            ],
            "answer_reuse_pair_refs": [
                _ref_payload(answer_reuse_pair_evidence_v2_ref(value)) for value in result.answer_reuse_pairs
            ],
            "cluster_refs": [
                _ref_payload(cross_item_safety_cluster_v2_ref(value)) for value in result.clusters
            ],
            "evaluated_prompt_foreign_set_count": (result.evaluated_prompt_foreign_set_count),
            "evaluated_attachment_foreign_set_count": (result.evaluated_attachment_foreign_set_count),
            "evaluated_answer_source_pair_count": (result.evaluated_answer_source_pair_count),
            "attachment_scan_count": result.attachment_scan_count,
            "visible_match_count": result.visible_match_count,
            "answer_reuse_pair_count": result.answer_reuse_pair_count,
            "cluster_count": result.cluster_count,
            "policy_version": result.policy_version,
        }
    )


def cross_item_safety_result_v2_ref(
    result: CrossItemSafetyResultV2,
) -> ObjectRef:
    return ObjectRef(
        object_type="cross-item-safety-result",
        object_id=result.cross_item_safety_result_id,
        object_version="v2",
        object_sha256=result.result_sha256,
    )


def validate_cross_item_safety_policy_v2_identity(
    policy: CrossItemSafetyPolicyV2,
) -> None:
    _require_current_identity(
        object_id=policy.cross_item_safety_policy_id,
        object_sha256=policy.policy_sha256,
        expected_prefix="cross-item-safety-policy",
        observed=cross_item_safety_policy_v2_carried_sha256(policy),
    )


def validate_attachment_cross_item_safety_scan_evidence_v2_identity(
    evidence: AttachmentCrossItemSafetyScanEvidenceV2,
) -> None:
    _require_hash_identity(
        object_sha256=evidence.evidence_sha256,
        observed=(attachment_cross_item_safety_scan_evidence_v2_carried_sha256(evidence)),
        label="attachment cross-item scan evidence",
    )


def validate_cross_item_visible_match_evidence_v2_identity(
    evidence: CrossItemVisibleMatchEvidenceV2,
) -> None:
    _require_current_identity(
        object_id=evidence.visible_match_evidence_id,
        object_sha256=evidence.evidence_sha256,
        expected_prefix="cross-item-visible-match-evidence",
        observed=cross_item_visible_match_evidence_v2_carried_sha256(evidence),
    )


def validate_answer_reuse_pair_evidence_v2_identity(
    pair: AnswerReusePairEvidenceV2,
) -> None:
    _require_current_identity(
        object_id=pair.answer_reuse_pair_id,
        object_sha256=pair.pair_sha256,
        expected_prefix="answer-reuse-pair-evidence",
        observed=answer_reuse_pair_evidence_v2_carried_sha256(pair),
    )


def validate_cross_item_safety_cluster_v2_identity(
    cluster: CrossItemSafetyClusterV2,
) -> None:
    _require_current_identity(
        object_id=cluster.cross_item_safety_cluster_id,
        object_sha256=cluster.cluster_sha256,
        expected_prefix="cross-item-safety-cluster",
        observed=cross_item_safety_cluster_v2_carried_sha256(cluster),
    )


def validate_cross_item_safety_result_v2_identity(
    result: CrossItemSafetyResultV2,
) -> None:
    _require_current_identity(
        object_id=result.cross_item_safety_result_id,
        object_sha256=result.result_sha256,
        expected_prefix="cross-item-safety-result",
        observed=cross_item_safety_result_v2_carried_sha256(result),
    )


def _validate_item_bindings(result: CrossItemSafetyResultV2) -> None:
    if (
        len(
            {
                len(result.item_ids),
                len(result.task_draft_refs),
                len(result.leakage_reference_set_refs),
                len(result.item_quality_result_refs),
            }
        )
        != 1
    ):
        raise ValueError("cross-item result Item binding lengths differ")
    _require_sorted_unique("result Item IDs", result.item_ids)
    for label, refs, object_type in (
        ("task draft refs", result.task_draft_refs, "task-draft"),
        (
            "leakage reference set refs",
            result.leakage_reference_set_refs,
            "prompt-leakage-reference-set",
        ),
        (
            "item quality result refs",
            result.item_quality_result_refs,
            "item-quality-compilation-result",
        ),
    ):
        _require_unique_refs(label, refs)
        for ref in refs:
            _require_ref(ref, object_type, "v2", label)
    if len(result.answer_source_item_ids) != len(result.answer_source_subject_refs):
        raise ValueError("answer source Item/subject inventories must align")
    answer_bindings = tuple(
        zip(
            result.answer_source_item_ids,
            result.answer_source_subject_refs,
            strict=True,
        )
    )
    if answer_bindings != tuple(
        sorted(
            answer_bindings,
            key=lambda value: (value[0], _ref_key(value[1])),
        )
    ):
        raise ValueError("answer source bindings must be canonical")
    _require_unique_refs(
        "answer source subjects",
        result.answer_source_subject_refs,
    )
    if any(item_id not in set(result.item_ids) for item_id in result.answer_source_item_ids):
        raise ValueError("answer source belongs to unknown Item")


def _validate_result_ownership(result: CrossItemSafetyResultV2) -> None:
    item_bindings = {
        item_id: (task_ref, reference_ref, quality_ref)
        for item_id, task_ref, reference_ref, quality_ref in zip(
            result.item_ids,
            result.task_draft_refs,
            result.leakage_reference_set_refs,
            result.item_quality_result_refs,
            strict=True,
        )
    }
    scan_by_ref = {
        attachment_cross_item_safety_scan_evidence_v2_ref(value): value
        for value in result.attachment_scan_evidence
    }
    _require_unique_refs(
        "attachment scan subjects",
        tuple(value.environment_artifact_ref for value in result.attachment_scan_evidence),
    )
    for scan_value in result.attachment_scan_evidence:
        binding = item_bindings.get(scan_value.target_item_id)
        if binding is None or scan_value.item_quality_result_ref != binding[2]:
            raise ValueError("attachment scan target ownership is invalid")
        expected_foreign = _sorted_refs(
            tuple(
                candidate[1]
                for item_id, candidate in item_bindings.items()
                if item_id != scan_value.target_item_id
            )
        )
        if scan_value.foreign_reference_set_refs != expected_foreign:
            raise ValueError("attachment scan foreign reference sets are not exact")
    visible_keys: set[
        tuple[
            CrossItemSafetyEvidenceKindV2,
            str,
            str,
            CrossItemSafetySurfaceKindV2,
            tuple[str, str, str, str],
            tuple[str, str, str, str],
            PromptLeakageCategoryV2,
        ]
    ] = set()
    for visible_value in result.visible_matches:
        source_binding = item_bindings.get(visible_value.source_item_id)
        target_binding = item_bindings.get(visible_value.target_item_id)
        if (
            source_binding is None
            or target_binding is None
            or visible_value.source_reference_set_ref != source_binding[1]
        ):
            raise ValueError("visible match Item ownership is invalid")
        if visible_value.target_surface is CrossItemSafetySurfaceKindV2.PROMPT:
            if visible_value.target_subject_ref != target_binding[0]:
                raise ValueError("prompt match target is not current")
        else:
            scan_ref = visible_value.attachment_scan_evidence_ref
            assert scan_ref is not None
            scan = scan_by_ref.get(scan_ref)
            if (
                scan is None
                or scan.target_item_id != visible_value.target_item_id
                or scan.environment_artifact_ref != visible_value.target_subject_ref
            ):
                raise ValueError("attachment match scan ownership is invalid")
        visible_key = (
            visible_value.evidence_kind,
            visible_value.source_item_id,
            visible_value.target_item_id,
            visible_value.target_surface,
            _ref_key(visible_value.target_subject_ref),
            _ref_key(visible_value.source_subject_ref),
            visible_value.category,
        )
        if visible_key in visible_keys:
            raise ValueError("duplicate visible match relationship")
        visible_keys.add(visible_key)
    pair_keys: set[
        tuple[
            str,
            str,
            tuple[str, str, str, str],
            tuple[str, str, str, str],
        ]
    ] = set()
    answer_source_bindings = set(
        zip(
            result.answer_source_item_ids,
            result.answer_source_subject_refs,
            strict=True,
        )
    )
    for reuse_value in result.answer_reuse_pairs:
        left_binding = item_bindings.get(reuse_value.left_item_id)
        right_binding = item_bindings.get(reuse_value.right_item_id)
        if (
            left_binding is None
            or right_binding is None
            or reuse_value.left_reference_set_ref != left_binding[1]
            or reuse_value.right_reference_set_ref != right_binding[1]
            or (
                reuse_value.left_item_id,
                reuse_value.left_source_subject_ref,
            )
            not in answer_source_bindings
            or (
                reuse_value.right_item_id,
                reuse_value.right_source_subject_ref,
            )
            not in answer_source_bindings
        ):
            raise ValueError("answer reuse pair ownership is invalid")
        pair_key = (
            reuse_value.left_item_id,
            reuse_value.right_item_id,
            _ref_key(reuse_value.left_source_subject_ref),
            _ref_key(reuse_value.right_source_subject_ref),
        )
        if pair_key in pair_keys:
            raise ValueError("duplicate answer reuse source pair")
        pair_keys.add(pair_key)


def _validate_result_counts(result: CrossItemSafetyResultV2) -> None:
    item_count = len(result.item_ids)
    expected_prompt = item_count * (item_count - 1)
    expected_attachment = len(result.attachment_scan_evidence) * (item_count - 1)
    answer_bindings = tuple(
        zip(
            result.answer_source_item_ids,
            result.answer_source_subject_refs,
            strict=True,
        )
    )
    expected_answer_pairs = sum(
        left_item != right_item
        for index, (left_item, _) in enumerate(answer_bindings)
        for right_item, _ in answer_bindings[index + 1 :]
    )
    if (
        result.evaluated_prompt_foreign_set_count != expected_prompt
        or result.evaluated_attachment_foreign_set_count != expected_attachment
        or result.evaluated_answer_source_pair_count != expected_answer_pairs
    ):
        raise ValueError("cross-item evaluated comparison counts are not exact")
    if (
        result.attachment_scan_count != len(result.attachment_scan_evidence)
        or result.visible_match_count != len(result.visible_matches)
        or result.answer_reuse_pair_count != len(result.answer_reuse_pairs)
        or result.cluster_count != len(result.clusters)
    ):
        raise ValueError("cross-item result inventory counts are not exact")


def _validate_result_clusters(result: CrossItemSafetyResultV2) -> None:
    direct_by_kind: dict[
        CrossItemSafetyClusterKindV2,
        tuple[tuple[ObjectRef, str, str], ...],
    ] = {
        CrossItemSafetyClusterKindV2.LEAKAGE: tuple(
            (
                cross_item_visible_match_evidence_v2_ref(value),
                value.source_item_id,
                value.target_item_id,
            )
            for value in result.visible_matches
            if value.evidence_kind is CrossItemSafetyEvidenceKindV2.LEAKAGE
        ),
        CrossItemSafetyClusterKindV2.CONTAMINATION: tuple(
            (
                cross_item_visible_match_evidence_v2_ref(value),
                value.source_item_id,
                value.target_item_id,
            )
            for value in result.visible_matches
            if value.evidence_kind is CrossItemSafetyEvidenceKindV2.CONTAMINATION
        ),
        CrossItemSafetyClusterKindV2.ANSWER_REUSE: tuple(
            (
                answer_reuse_pair_evidence_v2_ref(value),
                value.left_item_id,
                value.right_item_id,
            )
            for value in result.answer_reuse_pairs
        ),
    }
    expected: dict[
        CrossItemSafetyClusterKindV2,
        tuple[tuple[tuple[str, ...], tuple[ObjectRef, ...]], ...],
    ] = {kind: _edge_components(edges) for kind, edges in direct_by_kind.items()}
    actual: dict[
        CrossItemSafetyClusterKindV2,
        list[tuple[tuple[str, ...], tuple[ObjectRef, ...]]],
    ] = {kind: [] for kind in CrossItemSafetyClusterKindV2}
    for cluster in result.clusters:
        actual[cluster.cluster_kind].append(
            (
                cluster.member_item_ids,
                cluster.direct_evidence_refs,
            )
        )
    canonical_actual = {
        kind: tuple(
            sorted(
                values,
                key=lambda value: (
                    value[0],
                    tuple(_ref_key(ref) for ref in value[1]),
                ),
            )
        )
        for kind, values in actual.items()
    }
    if canonical_actual != expected:
        raise ValueError("cross-item safety clusters do not equal connected components")


def _edge_components(
    edges: tuple[tuple[ObjectRef, str, str], ...],
) -> tuple[tuple[tuple[str, ...], tuple[ObjectRef, ...]], ...]:
    if not edges:
        return ()
    adjacency: dict[str, set[str]] = {}
    for _, left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    remaining = set(adjacency)
    components: list[tuple[tuple[str, ...], tuple[ObjectRef, ...]]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        members: set[str] = set()
        while stack:
            item = stack.pop()
            if item in members:
                continue
            members.add(item)
            stack.extend(
                sorted(
                    adjacency.get(item, ()),
                    reverse=True,
                )
            )
        remaining.difference_update(members)
        components.append(
            (
                tuple(sorted(members)),
                _sorted_refs(
                    tuple(ref for ref, left, right in edges if left in members and right in members)
                ),
            )
        )
    return tuple(
        sorted(
            components,
            key=lambda value: (
                value[0],
                tuple(_ref_key(ref) for ref in value[1]),
            ),
        )
    )


def _cross_item_safety_result_refs(
    result: CrossItemSafetyResultV2,
) -> tuple[ObjectRef, ...]:
    return (
        result.resolved_job_work_graph_ref,
        result.policy_ref,
        *result.task_draft_refs,
        *result.leakage_reference_set_refs,
        *result.item_quality_result_refs,
        *result.answer_source_subject_refs,
        *(
            attachment_cross_item_safety_scan_evidence_v2_ref(value)
            for value in result.attachment_scan_evidence
        ),
        *(cross_item_visible_match_evidence_v2_ref(value) for value in result.visible_matches),
        *(answer_reuse_pair_evidence_v2_ref(value) for value in result.answer_reuse_pairs),
        *(cross_item_safety_cluster_v2_ref(value) for value in result.clusters),
    )


def _evidence_kind_for_category(
    category: PromptLeakageCategoryV2,
) -> CrossItemSafetyEvidenceKindV2:
    if category in _CONTAMINATION_CATEGORIES:
        return CrossItemSafetyEvidenceKindV2.CONTAMINATION
    if category in _LEAKAGE_CATEGORIES:
        return CrossItemSafetyEvidenceKindV2.LEAKAGE
    raise ValueError("unsupported cross-item safety category")


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


def _object_ref_from_facade(value: FacadeObjectRef) -> ObjectRef:
    return ObjectRef(
        object_type=value.object_type,
        object_id=value.object_id,
        object_version=value.object_version,
        object_sha256=value.object_sha256,
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
    keys = tuple(_ref_key(value) for value in values)
    _require_sorted_unique(label, keys)


def _require_unique_refs(
    label: str,
    values: tuple[ObjectRef, ...],
) -> None:
    keys = tuple(_ref_key(value) for value in values)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} must be unique")


def _require_nested_order(
    label: str,
    values: tuple[ObjectRef, ...],
) -> None:
    _require_sorted_unique_refs(label, values)


def _require_sorted_unique(
    label: str,
    values: tuple[Any, ...],
) -> None:
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{label} must be sorted and unique")


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
    _validate_identity(
        object_id=object_id,
        object_sha256=object_sha256,
        expected_prefix=expected_prefix,
        observed=observed,
    )


def _require_hash_identity(
    *,
    object_sha256: str,
    observed: str,
    label: str,
) -> None:
    if object_sha256 != observed:
        raise ValueError(f"{label} identity is stale")


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


def _sorted_nested[ValueT](
    values: tuple[ValueT, ...],
    ref_builder: Any,
) -> tuple[ValueT, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: _ref_key(ref_builder(value)),
        )
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
