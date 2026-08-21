from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.labeling_v2 import LabelDecisionValueV2

INDEPENDENT_LABEL_TEST_SET_POLICY_VERSION: Literal["independent-label-test-set/r8-01-v1"] = (
    "independent-label-test-set/r8-01-v1"
)

_SEMANTIC_CLASSES = (
    LabelDecisionValueV2.MATCH,
    LabelDecisionValueV2.NO_MATCH,
    LabelDecisionValueV2.ABSTAIN,
)


class LabelTestSetEvaluationKindV2(StrEnum):
    STRUCTURED = "STRUCTURED"
    SEMANTIC = "SEMANTIC"


class LabelTestSetSelectionAlgorithmV2(StrEnum):
    SHA256_CLASS_BALANCED_V1 = "SHA256_CLASS_BALANCED_V1"


class LabelTestSetAccessPurposeV2(StrEnum):
    R8_LABEL_STATISTICAL_EVALUATION = "R8_LABEL_STATISTICAL_EVALUATION"
    R8_LABEL_TEST_SET_INTEGRITY_AUDIT = "R8_LABEL_TEST_SET_INTEGRITY_AUDIT"


class LabelTestSetPartitionKindV2(StrEnum):
    TRAIN = "TRAIN"
    DEVELOPMENT = "DEVELOPMENT"


class LabelTestSetBalanceStatusV2(StrEnum):
    ACHIEVED = "ACHIEVED"
    SOURCE_POPULATION_LIMITED = "SOURCE_POPULATION_LIMITED"
    INSUFFICIENT = "INSUFFICIENT"


class LabelTestSetShortageReasonV2(StrEnum):
    CLASS_SHORTAGE = "CLASS_SHORTAGE"
    TOTAL_SHORTAGE = "TOTAL_SHORTAGE"


class LabelTestSetFreezeOutcomeV2(StrEnum):
    FROZEN = "FROZEN"
    STATISTICAL_GATE_PENDING = "STATISTICAL_GATE_PENDING"


class LabelTestSetAccessOutcomeV2(StrEnum):
    GRANTED = "GRANTED"
    DENIED = "DENIED"


class LabelTestSetAccessReasonV2(StrEnum):
    AUTHORIZED = "AUTHORIZED"
    PRINCIPAL_NOT_AUTHORIZED = "PRINCIPAL_NOT_AUTHORIZED"
    PURPOSE_NOT_AUTHORIZED = "PURPOSE_NOT_AUTHORIZED"
    DATASET_NOT_AUTHORIZED = "DATASET_NOT_AUTHORIZED"
    ACCESS_POLICY_STALE = "ACCESS_POLICY_STALE"
    MEMBER_LIMIT_EXCEEDED = "MEMBER_LIMIT_EXCEEDED"
    MATERIAL_INTEGRITY_FAILED = "MATERIAL_INTEGRITY_FAILED"


class IndependentLabelTestSetLabelPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/independent-label-test-set-label-policy/v2"] = (
        "eval-factory/independent-label-test-set-label-policy/v2"
    )
    label_spec_ref: ObjectRef
    evaluation_kind: LabelTestSetEvaluationKindV2
    structured_positive_minimum: int = Field(ge=0, le=1_000_000)
    structured_negative_minimum: int = Field(ge=0, le=1_000_000)
    semantic_total_minimum: int = Field(ge=0, le=1_000_000)
    semantic_classes: tuple[LabelDecisionValueV2, ...] = ()

    @field_validator("evaluation_kind", mode="before")
    @classmethod
    def parse_evaluation_kind(cls, value: object) -> LabelTestSetEvaluationKindV2:
        return _parse_enum(value, LabelTestSetEvaluationKindV2, "evaluation_kind")

    @field_validator("semantic_classes", mode="before")
    @classmethod
    def parse_semantic_classes(cls, value: object) -> tuple[LabelDecisionValueV2, ...]:
        if not isinstance(value, (tuple, list)):
            raise TypeError("semantic_classes must be a sequence")
        return tuple(_parse_enum(item, LabelDecisionValueV2, "semantic class") for item in value)

    @model_validator(mode="after")
    def validate_label_policy(self) -> Self:
        _require_ref(self.label_spec_ref, "label-spec", "v2", "label_spec_ref")
        if self.evaluation_kind is LabelTestSetEvaluationKindV2.STRUCTURED:
            if self.structured_positive_minimum < 50 or self.structured_negative_minimum < 50:
                raise ValueError("structured labels require at least 50 positive and 50 negative traces")
            if self.semantic_total_minimum != 0 or self.semantic_classes:
                raise ValueError("structured labels cannot carry semantic sample requirements")
        else:
            if self.structured_positive_minimum != 0 or self.structured_negative_minimum != 0:
                raise ValueError("semantic labels cannot carry structured sample requirements")
            if self.semantic_total_minimum < 100:
                raise ValueError("semantic labels require at least 100 traces")
            if self.semantic_classes != _SEMANTIC_CLASSES:
                raise ValueError("semantic classes must be MATCH, NO_MATCH, and ABSTAIN")
        return self

    @property
    def required_total(self) -> int:
        if self.evaluation_kind is LabelTestSetEvaluationKindV2.STRUCTURED:
            return self.structured_positive_minimum + self.structured_negative_minimum
        return self.semantic_total_minimum


class IndependentLabelTestSetPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/independent-label-test-set-policy/v2"] = (
        "eval-factory/independent-label-test-set-policy/v2"
    )
    policy_id: Identifier
    label_policies: tuple[IndependentLabelTestSetLabelPolicyV2, ...] = Field(min_length=1)
    selection_algorithm: Literal[LabelTestSetSelectionAlgorithmV2.SHA256_CLASS_BALANCED_V1] = (
        LabelTestSetSelectionAlgorithmV2.SHA256_CLASS_BALANCED_V1
    )
    selection_seed: Identifier
    trace_id_isolation: Literal[True] = True
    raw_hash_isolation: Literal[True] = True
    max_candidates: int = Field(ge=1, le=10_000_000)
    max_selected_members: int = Field(ge=1, le=10_000_000)
    max_private_bytes: int = Field(ge=2, le=5_000_000_000)
    policy_version: Literal["independent-label-test-set/r8-01-v1"] = INDEPENDENT_LABEL_TEST_SET_POLICY_VERSION
    policy_sha256: Sha256
    audit: ContractAudit

    @field_validator("selection_algorithm", mode="before")
    @classmethod
    def parse_selection_algorithm(cls, value: object) -> LabelTestSetSelectionAlgorithmV2:
        return _parse_enum(value, LabelTestSetSelectionAlgorithmV2, "selection_algorithm")

    @model_validator(mode="after")
    def validate_policy(self) -> Self:
        _require_sorted_unique_label_policies(self.label_policies)
        required = sum(item.required_total for item in self.label_policies)
        if self.max_selected_members < required:
            raise ValueError("max_selected_members cannot be below required label samples")
        if self.max_candidates < self.max_selected_members:
            raise ValueError("max_candidates cannot be below max_selected_members")
        _validate_audit(self.audit, (), "independent label test-set policy")
        _validate_identity(
            self.policy_id,
            self.policy_sha256,
            "independent-label-test-set-policy",
            independent_label_test_set_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        label_policies: tuple[IndependentLabelTestSetLabelPolicyV2, ...],
        selection_seed: str,
        max_candidates: int,
        max_selected_members: int,
        max_private_bytes: int,
        audit: ContractAudit,
    ) -> IndependentLabelTestSetPolicyV2:
        value = cls(
            policy_id="independent-label-test-set-policy://pending",
            label_policies=tuple(sorted(label_policies, key=lambda item: _ref_key(item.label_spec_ref))),
            selection_seed=selection_seed,
            max_candidates=max_candidates,
            max_selected_members=max_selected_members,
            max_private_bytes=max_private_bytes,
            policy_sha256="0" * 64,
            audit=_safe_audit(audit, ()),
        )
        return _finalize(
            value,
            "policy_id",
            "policy_sha256",
            "independent-label-test-set-policy",
            independent_label_test_set_policy_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return independent_label_test_set_policy_v2_ref(self)


class IndependentLabelTestSetPrincipalGrantV2(ContractModelV2):
    schema_version: Literal["eval-factory/independent-label-test-set-principal-grant/v2"] = (
        "eval-factory/independent-label-test-set-principal-grant/v2"
    )
    principal_ref: ObjectRef
    purposes: frozenset[LabelTestSetAccessPurposeV2] = Field(min_length=1)

    @field_validator("purposes", mode="before")
    @classmethod
    def parse_purposes(cls, value: object) -> frozenset[LabelTestSetAccessPurposeV2]:
        if not isinstance(value, (tuple, list, set, frozenset)):
            raise TypeError("purposes must be a collection")
        return frozenset(_parse_enum(item, LabelTestSetAccessPurposeV2, "purpose") for item in value)

    @model_validator(mode="after")
    def validate_grant(self) -> Self:
        _require_ref(self.principal_ref, "statistical-principal", "v1", "principal_ref")
        return self


class IndependentLabelTestSetAccessPolicyV2(ContractModelV2):
    schema_version: Literal["eval-factory/independent-label-test-set-access-policy/v2"] = (
        "eval-factory/independent-label-test-set-access-policy/v2"
    )
    access_policy_id: Identifier
    principal_grants: tuple[IndependentLabelTestSetPrincipalGrantV2, ...] = Field(min_length=1)
    max_members_per_read: int = Field(ge=1, le=10_000_000)
    policy_version: Literal["independent-label-test-set/r8-01-v1"] = INDEPENDENT_LABEL_TEST_SET_POLICY_VERSION
    access_policy_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_access_policy(self) -> Self:
        keys = tuple(_ref_key(item.principal_ref) for item in self.principal_grants)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("principal grants must be sorted and unique")
        refs = tuple(item.principal_ref for item in self.principal_grants)
        _validate_audit(self.audit, refs, "independent label test-set access policy")
        _validate_identity(
            self.access_policy_id,
            self.access_policy_sha256,
            "independent-label-test-set-access-policy",
            independent_label_test_set_access_policy_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        principal_grants: tuple[IndependentLabelTestSetPrincipalGrantV2, ...],
        max_members_per_read: int,
        audit: ContractAudit,
    ) -> IndependentLabelTestSetAccessPolicyV2:
        grants = tuple(sorted(principal_grants, key=lambda item: _ref_key(item.principal_ref)))
        refs = tuple(item.principal_ref for item in grants)
        value = cls(
            access_policy_id="independent-label-test-set-access-policy://pending",
            principal_grants=grants,
            max_members_per_read=max_members_per_read,
            access_policy_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "access_policy_id",
            "access_policy_sha256",
            "independent-label-test-set-access-policy",
            independent_label_test_set_access_policy_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return independent_label_test_set_access_policy_v2_ref(self)


class IndependentLabelPartitionManifestV2(ContractModelV2):
    schema_version: Literal["eval-factory/independent-label-partition-manifest/v2"] = (
        "eval-factory/independent-label-partition-manifest/v2"
    )
    partition_manifest_id: Identifier
    partition_kind: LabelTestSetPartitionKindV2
    private_material_ref: ObjectRef
    trace_count: int = Field(ge=0, le=10_000_000)
    unique_raw_hash_count: int = Field(ge=0, le=10_000_000)
    source_authority_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    policy_version: Literal["independent-label-test-set/r8-01-v1"] = INDEPENDENT_LABEL_TEST_SET_POLICY_VERSION
    partition_manifest_sha256: Sha256
    audit: ContractAudit

    @field_validator("partition_kind", mode="before")
    @classmethod
    def parse_partition_kind(cls, value: object) -> LabelTestSetPartitionKindV2:
        return _parse_enum(value, LabelTestSetPartitionKindV2, "partition_kind")

    @model_validator(mode="after")
    def validate_partition(self) -> Self:
        _require_ref(
            self.private_material_ref,
            "independent-label-partition-material",
            "private-v1",
            "private_material_ref",
        )
        if self.unique_raw_hash_count > self.trace_count:
            raise ValueError("unique raw hash count cannot exceed trace count")
        _require_sorted_unique_refs(self.source_authority_refs, "source_authority_refs")
        refs = (self.private_material_ref, *self.source_authority_refs)
        _validate_audit(self.audit, refs, "independent label partition manifest")
        _validate_identity(
            self.partition_manifest_id,
            self.partition_manifest_sha256,
            "independent-label-partition-manifest",
            independent_label_partition_manifest_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        partition_kind: LabelTestSetPartitionKindV2,
        private_material_ref: ObjectRef,
        trace_count: int,
        unique_raw_hash_count: int,
        source_authority_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> IndependentLabelPartitionManifestV2:
        refs = _sorted_refs(source_authority_refs)
        value = cls(
            partition_manifest_id="independent-label-partition-manifest://pending",
            partition_kind=partition_kind,
            private_material_ref=private_material_ref,
            trace_count=trace_count,
            unique_raw_hash_count=unique_raw_hash_count,
            source_authority_refs=refs,
            partition_manifest_sha256="0" * 64,
            audit=_safe_audit(audit, (private_material_ref, *refs)),
        )
        return _finalize(
            value,
            "partition_manifest_id",
            "partition_manifest_sha256",
            "independent-label-partition-manifest",
            independent_label_partition_manifest_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return independent_label_partition_manifest_v2_ref(self)


class LabelTestSetClassCountV2(ContractModelV2):
    schema_version: Literal["eval-factory/label-test-set-class-count/v2"] = (
        "eval-factory/label-test-set-class-count/v2"
    )
    decision: LabelDecisionValueV2
    eligible_count: int = Field(ge=0, le=10_000_000)
    selected_count: int = Field(ge=0, le=10_000_000)
    required_minimum: int = Field(ge=0, le=1_000_000)

    @field_validator("decision", mode="before")
    @classmethod
    def parse_decision(cls, value: object) -> LabelDecisionValueV2:
        return _parse_enum(value, LabelDecisionValueV2, "decision")

    @model_validator(mode="after")
    def validate_count(self) -> Self:
        if self.selected_count > self.eligible_count:
            raise ValueError("selected count cannot exceed eligible count")
        return self


class LabelTestSetLabelSummaryV2(ContractModelV2):
    schema_version: Literal["eval-factory/label-test-set-label-summary/v2"] = (
        "eval-factory/label-test-set-label-summary/v2"
    )
    label_spec_ref: ObjectRef
    evaluation_kind: LabelTestSetEvaluationKindV2
    class_counts: tuple[LabelTestSetClassCountV2, ...]
    required_total: int = Field(ge=1, le=1_000_000)
    eligible_total: int = Field(ge=0, le=10_000_000)
    selected_total: int = Field(ge=0, le=10_000_000)
    balance_status: LabelTestSetBalanceStatusV2

    @field_validator("evaluation_kind", mode="before")
    @classmethod
    def parse_evaluation_kind(cls, value: object) -> LabelTestSetEvaluationKindV2:
        return _parse_enum(value, LabelTestSetEvaluationKindV2, "evaluation_kind")

    @field_validator("balance_status", mode="before")
    @classmethod
    def parse_balance_status(cls, value: object) -> LabelTestSetBalanceStatusV2:
        return _parse_enum(value, LabelTestSetBalanceStatusV2, "balance_status")

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        _require_ref(self.label_spec_ref, "label-spec", "v2", "label_spec_ref")
        decisions = tuple(item.decision for item in self.class_counts)
        if decisions != _SEMANTIC_CLASSES:
            raise ValueError("class counts must be ordered MATCH, NO_MATCH, ABSTAIN")
        if self.eligible_total != sum(item.eligible_count for item in self.class_counts):
            raise ValueError("eligible total must equal class counts")
        if self.selected_total != sum(item.selected_count for item in self.class_counts):
            raise ValueError("selected total must equal class counts")
        by_decision = {item.decision: item for item in self.class_counts}
        if self.evaluation_kind is LabelTestSetEvaluationKindV2.STRUCTURED:
            match_count = by_decision[LabelDecisionValueV2.MATCH]
            no_match_count = by_decision[LabelDecisionValueV2.NO_MATCH]
            abstain_count = by_decision[LabelDecisionValueV2.ABSTAIN]
            if (
                match_count.required_minimum < 50
                or no_match_count.required_minimum < 50
                or abstain_count.required_minimum != 0
                or abstain_count.selected_count != 0
                or self.required_total != match_count.required_minimum + no_match_count.required_minimum
            ):
                raise ValueError("structured summary must preserve binary NFR-008 requirements")
        else:
            if self.required_total < 100:
                raise ValueError("semantic summary requires at least 100 traces")
            if any(item.required_minimum != 0 for item in self.class_counts):
                raise ValueError("semantic class minimums must remain source-population dependent")
        if self.selected_total < self.required_total:
            if self.balance_status is not LabelTestSetBalanceStatusV2.INSUFFICIENT:
                raise ValueError("insufficient summary requires INSUFFICIENT balance status")
        elif self.balance_status is LabelTestSetBalanceStatusV2.INSUFFICIENT:
            raise ValueError("sufficient summary cannot use INSUFFICIENT balance status")
        return self


class LabelTestSetShortageV2(ContractModelV2):
    schema_version: Literal["eval-factory/label-test-set-shortage/v2"] = (
        "eval-factory/label-test-set-shortage/v2"
    )
    label_spec_ref: ObjectRef
    decision: LabelDecisionValueV2 | None = None
    reason: LabelTestSetShortageReasonV2
    required_count: int = Field(ge=1, le=1_000_000)
    available_count: int = Field(ge=0, le=1_000_000)

    @field_validator("decision", mode="before")
    @classmethod
    def parse_decision(cls, value: object) -> LabelDecisionValueV2 | None:
        if value is None:
            return None
        return _parse_enum(value, LabelDecisionValueV2, "decision")

    @field_validator("reason", mode="before")
    @classmethod
    def parse_reason(cls, value: object) -> LabelTestSetShortageReasonV2:
        return _parse_enum(value, LabelTestSetShortageReasonV2, "reason")

    @model_validator(mode="after")
    def validate_shortage(self) -> Self:
        _require_ref(self.label_spec_ref, "label-spec", "v2", "label_spec_ref")
        if self.available_count >= self.required_count:
            raise ValueError("shortage requires available count below required count")
        if self.reason is LabelTestSetShortageReasonV2.CLASS_SHORTAGE and self.decision is None:
            raise ValueError("class shortage requires a decision")
        if self.reason is LabelTestSetShortageReasonV2.TOTAL_SHORTAGE and self.decision is not None:
            raise ValueError("total shortage cannot carry a decision")
        return self


class IndependentLabelTestSetManifestV2(ContractModelV2):
    schema_version: Literal["eval-factory/independent-label-test-set-manifest/v2"] = (
        "eval-factory/independent-label-test-set-manifest/v2"
    )
    dataset_id: Identifier
    dataset_series_id: Identifier
    dataset_version: str = Field(min_length=1, max_length=128)
    private_material_ref: ObjectRef
    policy_ref: ObjectRef
    access_policy_ref: ObjectRef
    train_partition_ref: ObjectRef
    development_partition_ref: ObjectRef
    candidate_pool_ref: ObjectRef
    label_spec_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    label_summaries: tuple[LabelTestSetLabelSummaryV2, ...] = Field(min_length=1)
    selected_member_count: int = Field(ge=1, le=10_000_000)
    unique_trace_count: int = Field(ge=1, le=10_000_000)
    unique_raw_hash_count: int = Field(ge=1, le=10_000_000)
    selection_algorithm: Literal[LabelTestSetSelectionAlgorithmV2.SHA256_CLASS_BALANCED_V1] = (
        LabelTestSetSelectionAlgorithmV2.SHA256_CLASS_BALANCED_V1
    )
    selection_seed: Identifier
    supersedes_dataset_ref: ObjectRef | None = None
    policy_version: Literal["independent-label-test-set/r8-01-v1"] = INDEPENDENT_LABEL_TEST_SET_POLICY_VERSION
    dataset_sha256: Sha256
    audit: ContractAudit

    @field_validator("selection_algorithm", mode="before")
    @classmethod
    def parse_selection_algorithm(cls, value: object) -> LabelTestSetSelectionAlgorithmV2:
        return _parse_enum(value, LabelTestSetSelectionAlgorithmV2, "selection_algorithm")

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        _require_ref(
            self.private_material_ref,
            "independent-label-test-set-material",
            "private-v1",
            "private_material_ref",
        )
        _require_ref(
            self.policy_ref,
            "independent-label-test-set-policy",
            "v2",
            "policy_ref",
        )
        _require_ref(
            self.access_policy_ref,
            "independent-label-test-set-access-policy",
            "v2",
            "access_policy_ref",
        )
        _require_ref(
            self.train_partition_ref,
            "independent-label-partition-manifest",
            "v2",
            "train_partition_ref",
        )
        _require_ref(
            self.development_partition_ref,
            "independent-label-partition-manifest",
            "v2",
            "development_partition_ref",
        )
        _require_ref(
            self.candidate_pool_ref,
            "independent-label-candidate-pool",
            "private-v1",
            "candidate_pool_ref",
        )
        if self.supersedes_dataset_ref is not None:
            _require_ref(
                self.supersedes_dataset_ref,
                "independent-label-test-set-manifest",
                "v2",
                "supersedes_dataset_ref",
            )
        _require_sorted_unique_summaries(self.label_summaries)
        expected_refs = tuple(item.label_spec_ref for item in self.label_summaries)
        if self.label_spec_refs != expected_refs:
            raise ValueError("label spec refs must exactly match label summaries")
        if any(item.selected_total < item.required_total for item in self.label_summaries):
            raise ValueError("frozen dataset requires every label sample minimum")
        if self.selected_member_count != sum(item.selected_total for item in self.label_summaries):
            raise ValueError("selected member count must equal label summaries")
        if self.unique_trace_count > self.selected_member_count:
            raise ValueError("unique trace count cannot exceed selected member count")
        if self.unique_raw_hash_count != self.unique_trace_count:
            raise ValueError("raw-hash isolation requires one unique hash per unique trace")
        refs = (
            self.private_material_ref,
            self.policy_ref,
            self.access_policy_ref,
            self.train_partition_ref,
            self.development_partition_ref,
            self.candidate_pool_ref,
            *self.label_spec_refs,
            *((self.supersedes_dataset_ref,) if self.supersedes_dataset_ref is not None else ()),
        )
        _validate_audit(self.audit, refs, "independent label test-set manifest")
        _validate_identity(
            self.dataset_id,
            self.dataset_sha256,
            "independent-label-test-set-manifest",
            independent_label_test_set_manifest_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        dataset_series_id: str,
        dataset_version: str,
        private_material_ref: ObjectRef,
        policy_ref: ObjectRef,
        access_policy_ref: ObjectRef,
        train_partition_ref: ObjectRef,
        development_partition_ref: ObjectRef,
        candidate_pool_ref: ObjectRef,
        label_summaries: tuple[LabelTestSetLabelSummaryV2, ...],
        selected_member_count: int,
        unique_trace_count: int,
        unique_raw_hash_count: int,
        selection_algorithm: Literal[LabelTestSetSelectionAlgorithmV2.SHA256_CLASS_BALANCED_V1],
        selection_seed: str,
        supersedes_dataset_ref: ObjectRef | None,
        audit: ContractAudit,
    ) -> IndependentLabelTestSetManifestV2:
        summaries = tuple(sorted(label_summaries, key=lambda item: _ref_key(item.label_spec_ref)))
        label_refs = tuple(item.label_spec_ref for item in summaries)
        refs = (
            private_material_ref,
            policy_ref,
            access_policy_ref,
            train_partition_ref,
            development_partition_ref,
            candidate_pool_ref,
            *label_refs,
            *((supersedes_dataset_ref,) if supersedes_dataset_ref is not None else ()),
        )
        value = cls(
            dataset_id="independent-label-test-set-manifest://pending",
            dataset_series_id=dataset_series_id,
            dataset_version=dataset_version,
            private_material_ref=private_material_ref,
            policy_ref=policy_ref,
            access_policy_ref=access_policy_ref,
            train_partition_ref=train_partition_ref,
            development_partition_ref=development_partition_ref,
            candidate_pool_ref=candidate_pool_ref,
            label_spec_refs=label_refs,
            label_summaries=summaries,
            selected_member_count=selected_member_count,
            unique_trace_count=unique_trace_count,
            unique_raw_hash_count=unique_raw_hash_count,
            selection_algorithm=selection_algorithm,
            selection_seed=selection_seed,
            supersedes_dataset_ref=supersedes_dataset_ref,
            dataset_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "dataset_id",
            "dataset_sha256",
            "independent-label-test-set-manifest",
            independent_label_test_set_manifest_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return independent_label_test_set_manifest_v2_ref(self)


class IndependentLabelTestSetFreezeRecordV2(ContractModelV2):
    schema_version: Literal["eval-factory/independent-label-test-set-freeze-record/v2"] = (
        "eval-factory/independent-label-test-set-freeze-record/v2"
    )
    freeze_id: Identifier
    dataset_series_id: Identifier
    requested_dataset_version: str = Field(min_length=1, max_length=128)
    outcome: LabelTestSetFreezeOutcomeV2
    candidate_pool_ref: ObjectRef
    policy_ref: ObjectRef
    access_policy_ref: ObjectRef
    train_partition_ref: ObjectRef
    development_partition_ref: ObjectRef
    label_summaries: tuple[LabelTestSetLabelSummaryV2, ...] = Field(min_length=1)
    shortages: tuple[LabelTestSetShortageV2, ...] = ()
    dataset_manifest_ref: ObjectRef | None = None
    supersedes_freeze_ref: ObjectRef | None = None
    policy_version: Literal["independent-label-test-set/r8-01-v1"] = INDEPENDENT_LABEL_TEST_SET_POLICY_VERSION
    freeze_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> LabelTestSetFreezeOutcomeV2:
        return _parse_enum(value, LabelTestSetFreezeOutcomeV2, "outcome")

    @model_validator(mode="after")
    def validate_freeze_record(self) -> Self:
        _require_ref(
            self.candidate_pool_ref,
            "independent-label-candidate-pool",
            "private-v1",
            "candidate_pool_ref",
        )
        _require_ref(
            self.policy_ref,
            "independent-label-test-set-policy",
            "v2",
            "policy_ref",
        )
        _require_ref(
            self.access_policy_ref,
            "independent-label-test-set-access-policy",
            "v2",
            "access_policy_ref",
        )
        for ref, name in (
            (self.train_partition_ref, "train_partition_ref"),
            (self.development_partition_ref, "development_partition_ref"),
        ):
            _require_ref(ref, "independent-label-partition-manifest", "v2", name)
        if self.dataset_manifest_ref is not None:
            _require_ref(
                self.dataset_manifest_ref,
                "independent-label-test-set-manifest",
                "v2",
                "dataset_manifest_ref",
            )
        if self.supersedes_freeze_ref is not None:
            _require_ref(
                self.supersedes_freeze_ref,
                "independent-label-test-set-freeze-record",
                "v2",
                "supersedes_freeze_ref",
            )
        _require_sorted_unique_summaries(self.label_summaries)
        shortage_keys = tuple(_shortage_key(item) for item in self.shortages)
        if shortage_keys != tuple(sorted(shortage_keys)) or len(shortage_keys) != len(set(shortage_keys)):
            raise ValueError("shortages must be sorted and unique")
        summary_refs = {item.label_spec_ref for item in self.label_summaries}
        if any(item.label_spec_ref not in summary_refs for item in self.shortages):
            raise ValueError("shortage label must exist in label summaries")
        if self.outcome is LabelTestSetFreezeOutcomeV2.FROZEN:
            if self.shortages:
                raise ValueError("FROZEN record cannot carry shortage")
            if self.dataset_manifest_ref is None:
                raise ValueError("FROZEN record requires dataset authority")
            if any(item.selected_total < item.required_total for item in self.label_summaries):
                raise ValueError("FROZEN record requires sufficient label summaries")
        else:
            if not self.shortages:
                raise ValueError("STATISTICAL_GATE_PENDING requires shortage")
            if self.dataset_manifest_ref is not None:
                raise ValueError("STATISTICAL_GATE_PENDING forbids dataset authority")
        refs = (
            self.candidate_pool_ref,
            self.policy_ref,
            self.access_policy_ref,
            self.train_partition_ref,
            self.development_partition_ref,
            *((self.dataset_manifest_ref,) if self.dataset_manifest_ref is not None else ()),
            *((self.supersedes_freeze_ref,) if self.supersedes_freeze_ref is not None else ()),
            *(item.label_spec_ref for item in self.label_summaries),
        )
        _validate_audit(self.audit, refs, "independent label test-set freeze record")
        _validate_identity(
            self.freeze_id,
            self.freeze_sha256,
            "independent-label-test-set-freeze-record",
            independent_label_test_set_freeze_record_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        dataset_series_id: str,
        requested_dataset_version: str,
        outcome: LabelTestSetFreezeOutcomeV2,
        candidate_pool_ref: ObjectRef,
        policy_ref: ObjectRef,
        access_policy_ref: ObjectRef,
        train_partition_ref: ObjectRef,
        development_partition_ref: ObjectRef,
        label_summaries: tuple[LabelTestSetLabelSummaryV2, ...],
        shortages: tuple[LabelTestSetShortageV2, ...],
        dataset_manifest_ref: ObjectRef | None,
        supersedes_freeze_ref: ObjectRef | None,
        audit: ContractAudit,
    ) -> IndependentLabelTestSetFreezeRecordV2:
        summaries = tuple(sorted(label_summaries, key=lambda item: _ref_key(item.label_spec_ref)))
        sorted_shortages = tuple(sorted(shortages, key=_shortage_key))
        refs = (
            candidate_pool_ref,
            policy_ref,
            access_policy_ref,
            train_partition_ref,
            development_partition_ref,
            *((dataset_manifest_ref,) if dataset_manifest_ref is not None else ()),
            *((supersedes_freeze_ref,) if supersedes_freeze_ref is not None else ()),
            *(item.label_spec_ref for item in summaries),
        )
        value = cls(
            freeze_id="independent-label-test-set-freeze-record://pending",
            dataset_series_id=dataset_series_id,
            requested_dataset_version=requested_dataset_version,
            outcome=outcome,
            candidate_pool_ref=candidate_pool_ref,
            policy_ref=policy_ref,
            access_policy_ref=access_policy_ref,
            train_partition_ref=train_partition_ref,
            development_partition_ref=development_partition_ref,
            label_summaries=summaries,
            shortages=sorted_shortages,
            dataset_manifest_ref=dataset_manifest_ref,
            supersedes_freeze_ref=supersedes_freeze_ref,
            freeze_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "freeze_id",
            "freeze_sha256",
            "independent-label-test-set-freeze-record",
            independent_label_test_set_freeze_record_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return independent_label_test_set_freeze_record_v2_ref(self)


class IndependentLabelTestSetFreezeResultV2(ContractModelV2):
    schema_version: Literal["eval-factory/independent-label-test-set-freeze-result/v2"] = (
        "eval-factory/independent-label-test-set-freeze-result/v2"
    )
    result_id: Identifier
    freeze_record: IndependentLabelTestSetFreezeRecordV2
    dataset_manifest: IndependentLabelTestSetManifestV2 | None = None
    policy_ref: ObjectRef
    access_policy_ref: ObjectRef
    partition_refs: tuple[ObjectRef, ObjectRef]
    policy_version: Literal["independent-label-test-set/r8-01-v1"] = INDEPENDENT_LABEL_TEST_SET_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        validate_independent_label_test_set_freeze_record_v2_identity(self.freeze_record)
        if self.dataset_manifest is not None:
            validate_independent_label_test_set_manifest_v2_identity(self.dataset_manifest)
        expected_partitions = (
            self.freeze_record.train_partition_ref,
            self.freeze_record.development_partition_ref,
        )
        if (
            self.policy_ref != self.freeze_record.policy_ref
            or self.access_policy_ref != self.freeze_record.access_policy_ref
            or self.partition_refs != expected_partitions
        ):
            raise ValueError("freeze result authority bindings are not exact")
        if self.freeze_record.outcome is LabelTestSetFreezeOutcomeV2.FROZEN:
            if (
                self.dataset_manifest is None
                or self.freeze_record.dataset_manifest_ref != self.dataset_manifest.to_ref()
                or self.dataset_manifest.dataset_series_id != self.freeze_record.dataset_series_id
                or self.dataset_manifest.dataset_version != self.freeze_record.requested_dataset_version
                or self.dataset_manifest.policy_ref != self.policy_ref
                or self.dataset_manifest.access_policy_ref != self.access_policy_ref
                or (
                    self.dataset_manifest.train_partition_ref,
                    self.dataset_manifest.development_partition_ref,
                )
                != self.partition_refs
                or self.dataset_manifest.label_summaries != self.freeze_record.label_summaries
            ):
                raise ValueError("FROZEN result dataset bindings are not exact")
        elif self.dataset_manifest is not None:
            raise ValueError("pending freeze result cannot carry dataset manifest")
        refs = (
            self.freeze_record.to_ref(),
            self.policy_ref,
            self.access_policy_ref,
            *self.partition_refs,
            *((self.dataset_manifest.to_ref(),) if self.dataset_manifest is not None else ()),
        )
        _validate_audit(self.audit, refs, "independent label test-set freeze result")
        _validate_identity(
            self.result_id,
            self.result_sha256,
            "independent-label-test-set-freeze-result",
            independent_label_test_set_freeze_result_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        freeze_record: IndependentLabelTestSetFreezeRecordV2,
        dataset_manifest: IndependentLabelTestSetManifestV2 | None,
        policy_ref: ObjectRef,
        access_policy_ref: ObjectRef,
        partition_refs: tuple[ObjectRef, ObjectRef],
        audit: ContractAudit,
    ) -> IndependentLabelTestSetFreezeResultV2:
        refs = (
            freeze_record.to_ref(),
            policy_ref,
            access_policy_ref,
            *partition_refs,
            *((dataset_manifest.to_ref(),) if dataset_manifest is not None else ()),
        )
        value = cls(
            result_id="independent-label-test-set-freeze-result://pending",
            freeze_record=freeze_record,
            dataset_manifest=dataset_manifest,
            policy_ref=policy_ref,
            access_policy_ref=access_policy_ref,
            partition_refs=partition_refs,
            result_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "result_id",
            "result_sha256",
            "independent-label-test-set-freeze-result",
            independent_label_test_set_freeze_result_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return independent_label_test_set_freeze_result_v2_ref(self)


class IndependentLabelTestSetAccessReceiptV2(ContractModelV2):
    schema_version: Literal["eval-factory/independent-label-test-set-access-receipt/v2"] = (
        "eval-factory/independent-label-test-set-access-receipt/v2"
    )
    receipt_id: Identifier
    dataset_manifest_ref: ObjectRef
    principal_ref: ObjectRef
    purpose: LabelTestSetAccessPurposeV2
    access_policy_ref: ObjectRef
    outcome: LabelTestSetAccessOutcomeV2
    reason: LabelTestSetAccessReasonV2
    returned_member_count: int = Field(ge=0, le=10_000_000)
    material_verified: bool
    policy_version: Literal["independent-label-test-set/r8-01-v1"] = INDEPENDENT_LABEL_TEST_SET_POLICY_VERSION
    receipt_sha256: Sha256
    audit: ContractAudit

    @field_validator("purpose", mode="before")
    @classmethod
    def parse_purpose(cls, value: object) -> LabelTestSetAccessPurposeV2:
        return _parse_enum(value, LabelTestSetAccessPurposeV2, "purpose")

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> LabelTestSetAccessOutcomeV2:
        return _parse_enum(value, LabelTestSetAccessOutcomeV2, "outcome")

    @field_validator("reason", mode="before")
    @classmethod
    def parse_reason(cls, value: object) -> LabelTestSetAccessReasonV2:
        return _parse_enum(value, LabelTestSetAccessReasonV2, "reason")

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        _require_ref(
            self.dataset_manifest_ref,
            "independent-label-test-set-manifest",
            "v2",
            "dataset_manifest_ref",
        )
        _require_ref(self.principal_ref, "statistical-principal", "v1", "principal_ref")
        _require_ref(
            self.access_policy_ref,
            "independent-label-test-set-access-policy",
            "v2",
            "access_policy_ref",
        )
        if self.outcome is LabelTestSetAccessOutcomeV2.GRANTED:
            if (
                self.reason is not LabelTestSetAccessReasonV2.AUTHORIZED
                or self.returned_member_count < 1
                or not self.material_verified
            ):
                raise ValueError("granted access requires verified authorized material")
        elif (
            self.reason is LabelTestSetAccessReasonV2.AUTHORIZED
            or self.returned_member_count != 0
            or self.material_verified
        ):
            raise ValueError("denied access cannot return or verify private material")
        refs = (self.dataset_manifest_ref, self.principal_ref, self.access_policy_ref)
        _validate_audit(self.audit, refs, "independent label test-set access receipt")
        _validate_identity(
            self.receipt_id,
            self.receipt_sha256,
            "independent-label-test-set-access-receipt",
            independent_label_test_set_access_receipt_v2_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        dataset_manifest_ref: ObjectRef,
        principal_ref: ObjectRef,
        purpose: LabelTestSetAccessPurposeV2,
        access_policy_ref: ObjectRef,
        outcome: LabelTestSetAccessOutcomeV2,
        reason: LabelTestSetAccessReasonV2,
        returned_member_count: int,
        material_verified: bool,
        audit: ContractAudit,
    ) -> IndependentLabelTestSetAccessReceiptV2:
        refs = (dataset_manifest_ref, principal_ref, access_policy_ref)
        value = cls(
            receipt_id="independent-label-test-set-access-receipt://pending",
            dataset_manifest_ref=dataset_manifest_ref,
            principal_ref=principal_ref,
            purpose=purpose,
            access_policy_ref=access_policy_ref,
            outcome=outcome,
            reason=reason,
            returned_member_count=returned_member_count,
            material_verified=material_verified,
            receipt_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "receipt_id",
            "receipt_sha256",
            "independent-label-test-set-access-receipt",
            independent_label_test_set_access_receipt_v2_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return independent_label_test_set_access_receipt_v2_ref(self)


def independent_label_test_set_policy_v2_carried_sha256(
    value: IndependentLabelTestSetPolicyV2,
) -> str:
    return _carried(value, {"policy_id", "policy_sha256", "audit"})


def independent_label_test_set_access_policy_v2_carried_sha256(
    value: IndependentLabelTestSetAccessPolicyV2,
) -> str:
    return _carried(value, {"access_policy_id", "access_policy_sha256", "audit"})


def independent_label_partition_manifest_v2_carried_sha256(
    value: IndependentLabelPartitionManifestV2,
) -> str:
    return _carried(
        value,
        {"partition_manifest_id", "partition_manifest_sha256", "audit"},
    )


def independent_label_test_set_manifest_v2_carried_sha256(
    value: IndependentLabelTestSetManifestV2,
) -> str:
    return _carried(value, {"dataset_id", "dataset_sha256", "audit"})


def independent_label_test_set_freeze_record_v2_carried_sha256(
    value: IndependentLabelTestSetFreezeRecordV2,
) -> str:
    return _carried(value, {"freeze_id", "freeze_sha256", "audit"})


def independent_label_test_set_freeze_result_v2_carried_sha256(
    value: IndependentLabelTestSetFreezeResultV2,
) -> str:
    return _carried(value, {"result_id", "result_sha256", "audit"})


def independent_label_test_set_access_receipt_v2_carried_sha256(
    value: IndependentLabelTestSetAccessReceiptV2,
) -> str:
    return _carried(value, {"receipt_id", "receipt_sha256", "audit"})


def independent_label_test_set_policy_v2_ref(
    value: IndependentLabelTestSetPolicyV2,
) -> ObjectRef:
    validate_independent_label_test_set_policy_v2_identity(value)
    return _ref("independent-label-test-set-policy", value.policy_id, value.policy_sha256)


def independent_label_test_set_access_policy_v2_ref(
    value: IndependentLabelTestSetAccessPolicyV2,
) -> ObjectRef:
    validate_independent_label_test_set_access_policy_v2_identity(value)
    return _ref(
        "independent-label-test-set-access-policy",
        value.access_policy_id,
        value.access_policy_sha256,
    )


def independent_label_partition_manifest_v2_ref(
    value: IndependentLabelPartitionManifestV2,
) -> ObjectRef:
    validate_independent_label_partition_manifest_v2_identity(value)
    return _ref(
        "independent-label-partition-manifest",
        value.partition_manifest_id,
        value.partition_manifest_sha256,
    )


def independent_label_test_set_manifest_v2_ref(
    value: IndependentLabelTestSetManifestV2,
) -> ObjectRef:
    validate_independent_label_test_set_manifest_v2_identity(value)
    return _ref("independent-label-test-set-manifest", value.dataset_id, value.dataset_sha256)


def independent_label_test_set_freeze_record_v2_ref(
    value: IndependentLabelTestSetFreezeRecordV2,
) -> ObjectRef:
    validate_independent_label_test_set_freeze_record_v2_identity(value)
    return _ref(
        "independent-label-test-set-freeze-record",
        value.freeze_id,
        value.freeze_sha256,
    )


def independent_label_test_set_freeze_result_v2_ref(
    value: IndependentLabelTestSetFreezeResultV2,
) -> ObjectRef:
    validate_independent_label_test_set_freeze_result_v2_identity(value)
    return _ref(
        "independent-label-test-set-freeze-result",
        value.result_id,
        value.result_sha256,
    )


def independent_label_test_set_access_receipt_v2_ref(
    value: IndependentLabelTestSetAccessReceiptV2,
) -> ObjectRef:
    validate_independent_label_test_set_access_receipt_v2_identity(value)
    return _ref(
        "independent-label-test-set-access-receipt",
        value.receipt_id,
        value.receipt_sha256,
    )


def validate_independent_label_test_set_policy_v2_identity(
    value: IndependentLabelTestSetPolicyV2,
) -> None:
    _validate_identity(
        value.policy_id,
        value.policy_sha256,
        "independent-label-test-set-policy",
        independent_label_test_set_policy_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_independent_label_test_set_access_policy_v2_identity(
    value: IndependentLabelTestSetAccessPolicyV2,
) -> None:
    _validate_identity(
        value.access_policy_id,
        value.access_policy_sha256,
        "independent-label-test-set-access-policy",
        independent_label_test_set_access_policy_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_independent_label_partition_manifest_v2_identity(
    value: IndependentLabelPartitionManifestV2,
) -> None:
    _validate_identity(
        value.partition_manifest_id,
        value.partition_manifest_sha256,
        "independent-label-partition-manifest",
        independent_label_partition_manifest_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_independent_label_test_set_manifest_v2_identity(
    value: IndependentLabelTestSetManifestV2,
) -> None:
    _validate_identity(
        value.dataset_id,
        value.dataset_sha256,
        "independent-label-test-set-manifest",
        independent_label_test_set_manifest_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_independent_label_test_set_freeze_record_v2_identity(
    value: IndependentLabelTestSetFreezeRecordV2,
) -> None:
    _validate_identity(
        value.freeze_id,
        value.freeze_sha256,
        "independent-label-test-set-freeze-record",
        independent_label_test_set_freeze_record_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_independent_label_test_set_freeze_result_v2_identity(
    value: IndependentLabelTestSetFreezeResultV2,
) -> None:
    _validate_identity(
        value.result_id,
        value.result_sha256,
        "independent-label-test-set-freeze-result",
        independent_label_test_set_freeze_result_v2_carried_sha256(value),
        allow_pending=False,
    )


def validate_independent_label_test_set_access_receipt_v2_identity(
    value: IndependentLabelTestSetAccessReceiptV2,
) -> None:
    _validate_identity(
        value.receipt_id,
        value.receipt_sha256,
        "independent-label-test-set-access-receipt",
        independent_label_test_set_access_receipt_v2_carried_sha256(value),
        allow_pending=False,
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


def _require_sorted_unique_label_policies(
    values: tuple[IndependentLabelTestSetLabelPolicyV2, ...],
) -> None:
    keys = tuple(_ref_key(item.label_spec_ref) for item in values)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError("label policies must be sorted and unique")


def _require_sorted_unique_summaries(
    values: tuple[LabelTestSetLabelSummaryV2, ...],
) -> None:
    keys = tuple(_ref_key(item.label_spec_ref) for item in values)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError("label summaries must be sorted and unique")


def _require_sorted_unique_refs(values: tuple[ObjectRef, ...], field_name: str) -> None:
    keys = tuple(_ref_key(item) for item in values)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} must be sorted and unique")


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(sorted(set(values), key=_ref_key))


def _shortage_key(
    value: LabelTestSetShortageV2,
) -> tuple[tuple[str, str, str, str], str, str]:
    return (
        _ref_key(value.label_spec_ref),
        "" if value.decision is None else value.decision.value,
        value.reason.value,
    )


def _ref(
    object_type: str,
    object_id: str,
    object_sha256: str,
) -> ObjectRef:
    if object_id.endswith("://pending") or object_sha256 == "0" * 64:
        raise ValueError(f"{object_type} identity is pending")
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version="v2",
        object_sha256=object_sha256,
    )


def _carried(value: ContractModelV2, exclude: set[str]) -> str:
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
    value: ObjectRef,
    object_type: str,
    object_version: str,
    field_name: str,
) -> None:
    if value.object_type != object_type or value.object_version != object_version:
        raise ValueError(f"{field_name} must reference {object_type} {object_version}")


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
