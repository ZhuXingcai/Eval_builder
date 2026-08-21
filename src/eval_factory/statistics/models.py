from __future__ import annotations

import hashlib
import json
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
from eval_factory.contracts.statistics_v2 import (
    INDEPENDENT_LABEL_TEST_SET_POLICY_VERSION,
    LabelTestSetAccessPurposeV2,
    LabelTestSetPartitionKindV2,
)

_DENIED_REF_MARKERS = frozenset(
    {
        "answer-bearing",
        "credential",
        "final-answer",
        "final-output",
        "grader-rule",
        "hidden-pass-condition",
        "private-reference",
        "quarantine",
        "raw-trace",
        "raw-traj",
        "secret",
        "trace-raw",
    }
)


class IndependentLabelReferenceCandidateV1(ContractModelV2):
    schema_version: Literal["eval-factory/independent-label-reference-candidate/private-v1"] = (
        "eval-factory/independent-label-reference-candidate/private-v1"
    )
    trace_envelope_ref: ObjectRef
    source_trace_id: Identifier
    raw_sha256: Sha256
    label_spec_ref: ObjectRef
    annotation_ref: ObjectRef
    annotation_contract_ref: ObjectRef
    expected_decision: LabelDecisionValueV2
    reference_state: Literal["REFERENCE_READY"]
    reference_policy_version: str = Field(min_length=1, max_length=128)

    @field_validator("expected_decision", mode="before")
    @classmethod
    def parse_expected_decision(cls, value: object) -> LabelDecisionValueV2:
        if isinstance(value, LabelDecisionValueV2):
            return value
        if isinstance(value, str):
            return LabelDecisionValueV2(value)
        raise TypeError("expected_decision must be a LabelDecisionValueV2")

    @model_validator(mode="after")
    def validate_candidate(self) -> Self:
        _require_ref(self.trace_envelope_ref, "trace-envelope", "v1", "trace_envelope_ref")
        _require_ref(self.label_spec_ref, "label-spec", "v2", "label_spec_ref")
        _require_ref(
            self.annotation_ref,
            "independent-label-annotation",
            "v1",
            "annotation_ref",
        )
        _require_ref(
            self.annotation_contract_ref,
            "annotation-contract-manifest",
            "v1",
            "annotation_contract_ref",
        )
        for ref in (
            self.trace_envelope_ref,
            self.label_spec_ref,
            self.annotation_ref,
            self.annotation_contract_ref,
        ):
            _reject_unsafe_ref(ref)
        return self


class IndependentLabelCandidatePoolV1(ContractModelV2):
    schema_version: Literal["eval-factory/independent-label-candidate-pool/private-v1"] = (
        "eval-factory/independent-label-candidate-pool/private-v1"
    )
    candidate_pool_id: Identifier
    source_population_ref: ObjectRef
    candidates: tuple[IndependentLabelReferenceCandidateV1, ...] = Field(min_length=1)
    policy_version: Literal["independent-label-test-set/r8-01-v1"] = INDEPENDENT_LABEL_TEST_SET_POLICY_VERSION
    candidate_pool_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_pool(self) -> Self:
        _require_ref(
            self.source_population_ref,
            "independent-label-source-population",
            "v1",
            "source_population_ref",
        )
        _reject_unsafe_ref(self.source_population_ref)
        keys = tuple(_candidate_key(item) for item in self.candidates)
        if keys != tuple(sorted(keys)):
            raise ValueError("candidate pool must be sorted")
        pair_keys = tuple(_candidate_pair_key(item) for item in self.candidates)
        if len(pair_keys) != len(set(pair_keys)):
            raise ValueError("candidate pool contains duplicate trace-label pair")
        refs = (
            self.source_population_ref,
            *(item.trace_envelope_ref for item in self.candidates),
            *(item.label_spec_ref for item in self.candidates),
            *(item.annotation_ref for item in self.candidates),
            *(item.annotation_contract_ref for item in self.candidates),
        )
        _validate_audit(self.audit, refs, "independent label candidate pool")
        _validate_identity(
            self.candidate_pool_id,
            self.candidate_pool_sha256,
            "independent-label-candidate-pool",
            independent_label_candidate_pool_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        candidates: tuple[IndependentLabelReferenceCandidateV1, ...],
        source_population_ref: ObjectRef,
        audit: ContractAudit,
    ) -> IndependentLabelCandidatePoolV1:
        ordered = tuple(sorted(candidates, key=_candidate_key))
        refs = (
            source_population_ref,
            *(item.trace_envelope_ref for item in ordered),
            *(item.label_spec_ref for item in ordered),
            *(item.annotation_ref for item in ordered),
            *(item.annotation_contract_ref for item in ordered),
        )
        value = cls(
            candidate_pool_id="independent-label-candidate-pool://pending",
            source_population_ref=source_population_ref,
            candidates=ordered,
            candidate_pool_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "candidate_pool_id",
            "candidate_pool_sha256",
            "independent-label-candidate-pool",
            independent_label_candidate_pool_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return independent_label_candidate_pool_v1_ref(self)


class IndependentLabelPartitionMemberV1(ContractModelV2):
    schema_version: Literal["eval-factory/independent-label-partition-member/private-v1"] = (
        "eval-factory/independent-label-partition-member/private-v1"
    )
    source_trace_id: Identifier
    raw_sha256: Sha256


class IndependentLabelPartitionMaterialV1(ContractModelV2):
    schema_version: Literal["eval-factory/independent-label-partition-material/private-v1"] = (
        "eval-factory/independent-label-partition-material/private-v1"
    )
    partition_material_id: Identifier
    partition_kind: LabelTestSetPartitionKindV2
    members: tuple[IndependentLabelPartitionMemberV1, ...]
    source_authority_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    policy_version: Literal["independent-label-test-set/r8-01-v1"] = INDEPENDENT_LABEL_TEST_SET_POLICY_VERSION
    partition_material_sha256: Sha256
    audit: ContractAudit

    @field_validator("partition_kind", mode="before")
    @classmethod
    def parse_partition_kind(cls, value: object) -> LabelTestSetPartitionKindV2:
        if isinstance(value, LabelTestSetPartitionKindV2):
            return value
        if isinstance(value, str):
            return LabelTestSetPartitionKindV2(value)
        raise TypeError("partition_kind must be a LabelTestSetPartitionKindV2")

    @model_validator(mode="after")
    def validate_material(self) -> Self:
        keys = tuple(_partition_member_key(item) for item in self.members)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("partition members must be sorted and unique")
        _require_one_to_one_trace_hash(self.members, "partition")
        _require_sorted_unique_refs(self.source_authority_refs, "source_authority_refs")
        for ref in self.source_authority_refs:
            _reject_unsafe_ref(ref)
        _validate_audit(
            self.audit,
            self.source_authority_refs,
            "independent label partition material",
        )
        _validate_identity(
            self.partition_material_id,
            self.partition_material_sha256,
            "independent-label-partition-material",
            independent_label_partition_material_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        partition_kind: LabelTestSetPartitionKindV2,
        members: tuple[tuple[str, str], ...],
        source_authority_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
    ) -> IndependentLabelPartitionMaterialV1:
        normalized_members = tuple(
            sorted(
                (
                    IndependentLabelPartitionMemberV1(
                        source_trace_id=source_trace_id,
                        raw_sha256=raw_sha256,
                    )
                    for source_trace_id, raw_sha256 in members
                ),
                key=_partition_member_key,
            )
        )
        refs = _sorted_refs(source_authority_refs)
        value = cls(
            partition_material_id="independent-label-partition-material://pending",
            partition_kind=partition_kind,
            members=normalized_members,
            source_authority_refs=refs,
            partition_material_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "partition_material_id",
            "partition_material_sha256",
            "independent-label-partition-material",
            independent_label_partition_material_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return independent_label_partition_material_v1_ref(self)


class IndependentLabelTestSetMaterialV1(ContractModelV2):
    schema_version: Literal["eval-factory/independent-label-test-set-material/private-v1"] = (
        "eval-factory/independent-label-test-set-material/private-v1"
    )
    material_id: Identifier
    dataset_series_id: Identifier
    dataset_version: str = Field(min_length=1, max_length=128)
    members: tuple[IndependentLabelReferenceCandidateV1, ...] = Field(min_length=1)
    candidate_pool_ref: ObjectRef
    train_partition_ref: ObjectRef
    development_partition_ref: ObjectRef
    policy_ref: ObjectRef
    access_policy_ref: ObjectRef
    policy_version: Literal["independent-label-test-set/r8-01-v1"] = INDEPENDENT_LABEL_TEST_SET_POLICY_VERSION
    material_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_material(self) -> Self:
        keys = tuple(_candidate_key(item) for item in self.members)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("test-set members must be sorted and unique")
        _require_ref(
            self.candidate_pool_ref,
            "independent-label-candidate-pool",
            "private-v1",
            "candidate_pool_ref",
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
        _require_candidate_trace_hash_mapping(self.members, "test-set")
        refs = (
            self.candidate_pool_ref,
            self.train_partition_ref,
            self.development_partition_ref,
            self.policy_ref,
            self.access_policy_ref,
            *(item.trace_envelope_ref for item in self.members),
            *(item.label_spec_ref for item in self.members),
            *(item.annotation_ref for item in self.members),
        )
        _validate_audit(self.audit, refs, "independent label test-set material")
        _validate_identity(
            self.material_id,
            self.material_sha256,
            "independent-label-test-set-material",
            independent_label_test_set_material_v1_carried_sha256(self),
        )
        return self

    @classmethod
    def create(
        cls,
        *,
        dataset_series_id: str,
        dataset_version: str,
        members: tuple[IndependentLabelReferenceCandidateV1, ...],
        candidate_pool_ref: ObjectRef,
        train_partition_ref: ObjectRef,
        development_partition_ref: ObjectRef,
        policy_ref: ObjectRef,
        access_policy_ref: ObjectRef,
        audit: ContractAudit,
    ) -> IndependentLabelTestSetMaterialV1:
        ordered = tuple(sorted(members, key=_candidate_key))
        refs = (
            candidate_pool_ref,
            train_partition_ref,
            development_partition_ref,
            policy_ref,
            access_policy_ref,
            *(item.trace_envelope_ref for item in ordered),
            *(item.label_spec_ref for item in ordered),
            *(item.annotation_ref for item in ordered),
        )
        value = cls(
            material_id="independent-label-test-set-material://pending",
            dataset_series_id=dataset_series_id,
            dataset_version=dataset_version,
            members=ordered,
            candidate_pool_ref=candidate_pool_ref,
            train_partition_ref=train_partition_ref,
            development_partition_ref=development_partition_ref,
            policy_ref=policy_ref,
            access_policy_ref=access_policy_ref,
            material_sha256="0" * 64,
            audit=_safe_audit(audit, refs),
        )
        return _finalize(
            value,
            "material_id",
            "material_sha256",
            "independent-label-test-set-material",
            independent_label_test_set_material_v1_carried_sha256(value),
        )

    def to_ref(self) -> ObjectRef:
        return independent_label_test_set_material_v1_ref(self)


class TrustedIndependentLabelTestSetPrincipalV1(ContractModelV2):
    schema_version: Literal["eval-factory/trusted-independent-label-test-set-principal/private-v1"] = (
        "eval-factory/trusted-independent-label-test-set-principal/private-v1"
    )
    principal_ref: ObjectRef
    allowed_purposes: frozenset[LabelTestSetAccessPurposeV2] = Field(min_length=1)
    allowed_dataset_refs: frozenset[ObjectRef] = Field(min_length=1)
    max_members: int = Field(ge=1, le=10_000_000)
    audit_case_ref: ObjectRef | None = None

    @field_validator("allowed_purposes", mode="before")
    @classmethod
    def parse_purposes(cls, value: object) -> frozenset[LabelTestSetAccessPurposeV2]:
        if not isinstance(value, (tuple, list, set, frozenset)):
            raise TypeError("allowed_purposes must be a collection")
        result: set[LabelTestSetAccessPurposeV2] = set()
        for item in value:
            if isinstance(item, LabelTestSetAccessPurposeV2):
                result.add(item)
            elif isinstance(item, str):
                result.add(LabelTestSetAccessPurposeV2(item))
            else:
                raise TypeError("allowed purpose must be a LabelTestSetAccessPurposeV2")
        return frozenset(result)

    @field_validator("allowed_dataset_refs", mode="before")
    @classmethod
    def parse_dataset_refs(cls, value: object) -> frozenset[ObjectRef]:
        if not isinstance(value, (tuple, list, set, frozenset)):
            raise TypeError("allowed_dataset_refs must be a collection")
        return frozenset(
            item if isinstance(item, ObjectRef) else ObjectRef.model_validate(item) for item in value
        )

    @model_validator(mode="after")
    def validate_principal(self) -> Self:
        _require_ref(self.principal_ref, "statistical-principal", "v1", "principal_ref")
        for ref in self.allowed_dataset_refs:
            _require_ref(
                ref,
                "independent-label-test-set-manifest",
                "v2",
                "allowed_dataset_refs",
            )
        if (
            LabelTestSetAccessPurposeV2.R8_LABEL_TEST_SET_INTEGRITY_AUDIT in self.allowed_purposes
            and self.audit_case_ref is None
        ):
            raise ValueError("integrity audit principal requires audit_case_ref")
        if self.audit_case_ref is not None:
            _require_ref(self.audit_case_ref, "audit-case", "v1", "audit_case_ref")
        return self


class IndependentLabelTestSetAccessRequestV1(ContractModelV2):
    schema_version: Literal["eval-factory/independent-label-test-set-access-request/private-v1"] = (
        "eval-factory/independent-label-test-set-access-request/private-v1"
    )
    dataset_manifest_ref: ObjectRef
    access_policy_ref: ObjectRef
    purpose: LabelTestSetAccessPurposeV2
    max_members: int = Field(ge=1, le=10_000_000)
    idempotency_key: Identifier

    @field_validator("purpose", mode="before")
    @classmethod
    def parse_purpose(cls, value: object) -> LabelTestSetAccessPurposeV2:
        if isinstance(value, LabelTestSetAccessPurposeV2):
            return value
        if isinstance(value, str):
            return LabelTestSetAccessPurposeV2(value)
        raise TypeError("purpose must be a LabelTestSetAccessPurposeV2")

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        _require_ref(
            self.dataset_manifest_ref,
            "independent-label-test-set-manifest",
            "v2",
            "dataset_manifest_ref",
        )
        _require_ref(
            self.access_policy_ref,
            "independent-label-test-set-access-policy",
            "v2",
            "access_policy_ref",
        )
        return self


def independent_label_candidate_pool_v1_carried_sha256(
    value: IndependentLabelCandidatePoolV1,
) -> str:
    return _carried(value, {"candidate_pool_id", "candidate_pool_sha256", "audit"})


def independent_label_partition_material_v1_carried_sha256(
    value: IndependentLabelPartitionMaterialV1,
) -> str:
    return _carried(value, {"partition_material_id", "partition_material_sha256", "audit"})


def independent_label_test_set_material_v1_carried_sha256(
    value: IndependentLabelTestSetMaterialV1,
) -> str:
    return _carried(value, {"material_id", "material_sha256", "audit"})


def independent_label_candidate_pool_v1_ref(
    value: IndependentLabelCandidatePoolV1,
) -> ObjectRef:
    _validate_identity(
        value.candidate_pool_id,
        value.candidate_pool_sha256,
        "independent-label-candidate-pool",
        independent_label_candidate_pool_v1_carried_sha256(value),
        allow_pending=False,
    )
    return ObjectRef(
        object_type="independent-label-candidate-pool",
        object_id=value.candidate_pool_id,
        object_version="private-v1",
        object_sha256=value.candidate_pool_sha256,
    )


def independent_label_partition_material_v1_ref(
    value: IndependentLabelPartitionMaterialV1,
) -> ObjectRef:
    _validate_identity(
        value.partition_material_id,
        value.partition_material_sha256,
        "independent-label-partition-material",
        independent_label_partition_material_v1_carried_sha256(value),
        allow_pending=False,
    )
    return ObjectRef(
        object_type="independent-label-partition-material",
        object_id=value.partition_material_id,
        object_version="private-v1",
        object_sha256=value.partition_material_sha256,
    )


def independent_label_test_set_material_v1_ref(
    value: IndependentLabelTestSetMaterialV1,
) -> ObjectRef:
    _validate_identity(
        value.material_id,
        value.material_sha256,
        "independent-label-test-set-material",
        independent_label_test_set_material_v1_carried_sha256(value),
        allow_pending=False,
    )
    return ObjectRef(
        object_type="independent-label-test-set-material",
        object_id=value.material_id,
        object_version="private-v1",
        object_sha256=value.material_sha256,
    )


def validate_independent_label_candidate_pool_v1_identity(
    value: IndependentLabelCandidatePoolV1,
) -> None:
    independent_label_candidate_pool_v1_ref(value)


def validate_independent_label_partition_material_v1_identity(
    value: IndependentLabelPartitionMaterialV1,
) -> None:
    independent_label_partition_material_v1_ref(value)


def validate_independent_label_test_set_material_v1_identity(
    value: IndependentLabelTestSetMaterialV1,
) -> None:
    independent_label_test_set_material_v1_ref(value)


def _candidate_key(
    value: IndependentLabelReferenceCandidateV1,
) -> tuple[tuple[str, str, str, str], str, str, tuple[str, str, str, str]]:
    return (
        _ref_key(value.label_spec_ref),
        value.expected_decision.value,
        value.source_trace_id,
        _ref_key(value.annotation_ref),
    )


def _candidate_pair_key(
    value: IndependentLabelReferenceCandidateV1,
) -> tuple[str, tuple[str, str, str, str]]:
    return (value.source_trace_id, _ref_key(value.label_spec_ref))


def _partition_member_key(
    value: IndependentLabelPartitionMemberV1,
) -> tuple[str, str]:
    return (value.source_trace_id, value.raw_sha256)


def _require_one_to_one_trace_hash(
    values: tuple[IndependentLabelPartitionMemberV1, ...],
    label: str,
) -> None:
    trace_to_hash: dict[str, str] = {}
    hash_to_trace: dict[str, str] = {}
    for value in values:
        prior_hash = trace_to_hash.setdefault(value.source_trace_id, value.raw_sha256)
        prior_trace = hash_to_trace.setdefault(value.raw_sha256, value.source_trace_id)
        if prior_hash != value.raw_sha256 or prior_trace != value.source_trace_id:
            raise ValueError(f"{label} trace IDs and raw hashes must be one-to-one")


def _require_candidate_trace_hash_mapping(
    values: tuple[IndependentLabelReferenceCandidateV1, ...],
    label: str,
) -> None:
    trace_to_hash: dict[str, str] = {}
    hash_to_trace: dict[str, str] = {}
    for value in values:
        prior_hash = trace_to_hash.setdefault(value.source_trace_id, value.raw_sha256)
        prior_trace = hash_to_trace.setdefault(value.raw_sha256, value.source_trace_id)
        if prior_hash != value.raw_sha256 or prior_trace != value.source_trace_id:
            raise ValueError(f"{label} trace IDs and raw hashes must be one-to-one")


def _reject_unsafe_ref(value: ObjectRef) -> None:
    normalized = f"{value.object_type}:{value.object_id}".casefold().replace("_", "-")
    if any(marker in normalized for marker in _DENIED_REF_MARKERS):
        raise ValueError("independent label material contains unsafe reference")


def _require_sorted_unique_refs(values: tuple[ObjectRef, ...], field_name: str) -> None:
    keys = tuple(_ref_key(item) for item in values)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} must be sorted and unique")


def _sorted_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(sorted(set(values), key=_ref_key))


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def _require_ref(
    value: ObjectRef,
    object_type: str,
    object_version: str,
    field_name: str,
) -> None:
    if value.object_type != object_type or value.object_version != object_version:
        raise ValueError(f"{field_name} must reference {object_type} {object_version}")


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
