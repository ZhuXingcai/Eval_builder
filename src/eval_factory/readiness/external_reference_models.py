from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2
from eval_factory.contracts.external_evidence_v2 import (
    EXTERNAL_EVIDENCE_POLICY_VERSION,
)
from eval_factory.contracts.labeling_v2 import LabelDecisionValueV2
from eval_factory.statistics.models import (
    IndependentLabelReferenceCandidateV1,
)


class ExternalReferenceAuthorKindV1(StrEnum):
    DETERMINISTIC_SERVICE = "DETERMINISTIC_SERVICE"
    SEMANTIC_AGENT = "SEMANTIC_AGENT"
    IMPORTED = "IMPORTED"


class ExternalReferenceStateV1(StrEnum):
    REFERENCE_READY = "REFERENCE_READY"
    ABSTAINED = "ABSTAINED"
    BLOCKED = "BLOCKED"


class ExternalReferenceRecordV1(ContractModelV2):
    schema_version: Literal["eval-factory/external-reference-record/private-v1"] = (
        "eval-factory/external-reference-record/private-v1"
    )
    reference_record_id: Identifier
    source_trace_id: Identifier
    raw_sha256: Sha256
    trace_envelope_ref: ObjectRef
    label_spec_ref: ObjectRef
    label_name: Identifier
    annotation_contract_ref: ObjectRef
    reference_policy_ref: ObjectRef
    expected_decision: LabelDecisionValueV2
    evidence_refs: tuple[ObjectRef, ...]
    structured_capability_complete: bool
    author_kind: ExternalReferenceAuthorKindV1
    reference_state: ExternalReferenceStateV1
    rule_version: str | None = Field(default=None, min_length=1, max_length=128)
    model_profile: Identifier | None = None
    prompt_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    policy_version: Literal["external-evidence/r8-10-v1"] = EXTERNAL_EVIDENCE_POLICY_VERSION
    reference_record_sha256: Sha256
    audit: ContractAudit

    @classmethod
    def create(
        cls,
        *,
        source_trace_id: str,
        raw_sha256: str,
        trace_envelope_ref: ObjectRef,
        label_spec_ref: ObjectRef,
        label_name: str,
        annotation_contract_ref: ObjectRef,
        reference_policy_ref: ObjectRef,
        expected_decision: LabelDecisionValueV2,
        evidence_refs: tuple[ObjectRef, ...],
        structured_capability_complete: bool,
        author_kind: ExternalReferenceAuthorKindV1,
        reference_state: ExternalReferenceStateV1,
        rule_version: str | None,
        model_profile: str | None,
        prompt_version: str | None,
        audit: ContractAudit,
    ) -> ExternalReferenceRecordV1:
        refs = _sorted_refs(
            (
                trace_envelope_ref,
                label_spec_ref,
                annotation_contract_ref,
                reference_policy_ref,
                *evidence_refs,
            )
        )
        ordered_evidence = _sorted_refs(evidence_refs)
        safe_audit = audit.model_copy(update={"input_refs": refs})
        value = cls.model_construct(
            reference_record_id="external-reference-record://pending",
            source_trace_id=source_trace_id,
            raw_sha256=raw_sha256,
            trace_envelope_ref=trace_envelope_ref,
            label_spec_ref=label_spec_ref,
            label_name=label_name,
            annotation_contract_ref=annotation_contract_ref,
            reference_policy_ref=reference_policy_ref,
            expected_decision=expected_decision,
            evidence_refs=ordered_evidence,
            structured_capability_complete=structured_capability_complete,
            author_kind=author_kind,
            reference_state=reference_state,
            rule_version=rule_version,
            model_profile=model_profile,
            prompt_version=prompt_version,
            reference_record_sha256="0" * 64,
            audit=safe_audit,
        )
        digest = _carried_sha256(
            value,
            exclude={
                "reference_record_id",
                "reference_record_sha256",
                "audit",
            },
        )
        return cls(
            reference_record_id=(f"external-reference-record://sha256/{digest}"),
            source_trace_id=source_trace_id,
            raw_sha256=raw_sha256,
            trace_envelope_ref=trace_envelope_ref,
            label_spec_ref=label_spec_ref,
            label_name=label_name,
            annotation_contract_ref=annotation_contract_ref,
            reference_policy_ref=reference_policy_ref,
            expected_decision=expected_decision,
            evidence_refs=ordered_evidence,
            structured_capability_complete=structured_capability_complete,
            author_kind=author_kind,
            reference_state=reference_state,
            rule_version=rule_version,
            model_profile=model_profile,
            prompt_version=prompt_version,
            reference_record_sha256=digest,
            audit=safe_audit,
        )

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        _require_ref(
            self.trace_envelope_ref,
            "trace-envelope",
            "v1",
            "trace_envelope_ref",
        )
        _require_ref(
            self.label_spec_ref,
            "label-spec",
            "v2",
            "label_spec_ref",
        )
        _require_ref(
            self.annotation_contract_ref,
            "annotation-contract-manifest",
            "v1",
            "annotation_contract_ref",
        )
        _require_ref(
            self.reference_policy_ref,
            "external-reference-authoring-policy",
            "v2",
            "reference_policy_ref",
        )
        if self.evidence_refs != _sorted_refs(self.evidence_refs):
            raise ValueError("external reference evidence refs must be sorted")
        if self.expected_decision is LabelDecisionValueV2.MATCH and not self.evidence_refs:
            raise ValueError("reference MATCH requires evidence")
        if self.expected_decision is LabelDecisionValueV2.NO_MATCH and (
            not self.structured_capability_complete or not self.evidence_refs
        ):
            raise ValueError("reference NO_MATCH requires complete negative evidence")
        if (
            self.expected_decision is LabelDecisionValueV2.ABSTAIN
            and self.reference_state is ExternalReferenceStateV1.REFERENCE_READY
        ):
            raise ValueError("ambiguous reference cannot be marked ready")
        if self.author_kind is ExternalReferenceAuthorKindV1.DETERMINISTIC_SERVICE and (
            self.rule_version is None or self.model_profile is not None or self.prompt_version is not None
        ):
            raise ValueError("deterministic reference metadata is invalid")
        if self.author_kind is ExternalReferenceAuthorKindV1.SEMANTIC_AGENT and (
            self.model_profile is None or self.prompt_version is None or self.rule_version is not None
        ):
            raise ValueError("semantic reference metadata is incomplete")
        refs = _sorted_refs(
            (
                self.trace_envelope_ref,
                self.label_spec_ref,
                self.annotation_contract_ref,
                self.reference_policy_ref,
                *self.evidence_refs,
            )
        )
        if self.audit.input_refs != refs:
            raise ValueError("external reference audit refs are stale")
        digest = _carried_sha256(
            self,
            exclude={
                "reference_record_id",
                "reference_record_sha256",
                "audit",
            },
        )
        if (
            self.reference_record_sha256 != digest
            or self.reference_record_id != f"external-reference-record://sha256/{digest}"
        ):
            raise ValueError("external reference identity is stale")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="external-reference-record",
            object_id=self.reference_record_id,
            object_version="private-v1",
            object_sha256=self.reference_record_sha256,
        )

    def annotation_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="independent-label-annotation",
            object_id=(f"independent-label-annotation://sha256/{self.reference_record_sha256}"),
            object_version="v1",
            object_sha256=self.reference_record_sha256,
        )

    def to_r8_candidate(self) -> IndependentLabelReferenceCandidateV1:
        if self.reference_state is not ExternalReferenceStateV1.REFERENCE_READY:
            raise ValueError("only ready references can enter R8-01")
        return IndependentLabelReferenceCandidateV1(
            trace_envelope_ref=self.trace_envelope_ref,
            source_trace_id=self.source_trace_id,
            raw_sha256=self.raw_sha256,
            label_spec_ref=self.label_spec_ref,
            annotation_ref=self.annotation_ref(),
            annotation_contract_ref=self.annotation_contract_ref,
            expected_decision=self.expected_decision,
            reference_state="REFERENCE_READY",
            reference_policy_version=self.policy_version,
        )


class ExternalReferenceSetV1(ContractModelV2):
    schema_version: Literal["eval-factory/external-reference-set/private-v1"] = (
        "eval-factory/external-reference-set/private-v1"
    )
    reference_set_id: Identifier
    source_population_ref: ObjectRef
    label_spec_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=256)
    reference_policy_ref: ObjectRef
    records: tuple[ExternalReferenceRecordV1, ...] = Field(
        min_length=1,
        max_length=30_000_000,
    )
    policy_version: Literal["external-evidence/r8-10-v1"] = EXTERNAL_EVIDENCE_POLICY_VERSION
    reference_set_sha256: Sha256
    audit: ContractAudit

    @classmethod
    def create(
        cls,
        *,
        source_population_ref: ObjectRef,
        label_spec_refs: tuple[ObjectRef, ...],
        reference_policy_ref: ObjectRef,
        records: tuple[ExternalReferenceRecordV1, ...],
        audit: ContractAudit,
    ) -> ExternalReferenceSetV1:
        ordered_records = tuple(sorted(records, key=_record_key))
        labels = _sorted_refs(label_spec_refs)
        refs = _sorted_refs(
            (
                source_population_ref,
                reference_policy_ref,
                *labels,
                *(record.to_ref() for record in ordered_records),
            )
        )
        safe_audit = audit.model_copy(update={"input_refs": refs})
        value = cls.model_construct(
            reference_set_id="external-reference-set://pending",
            source_population_ref=source_population_ref,
            label_spec_refs=labels,
            reference_policy_ref=reference_policy_ref,
            records=ordered_records,
            reference_set_sha256="0" * 64,
            audit=safe_audit,
        )
        digest = _carried_sha256(
            value,
            exclude={"reference_set_id", "reference_set_sha256", "audit"},
        )
        return cls(
            reference_set_id=f"external-reference-set://sha256/{digest}",
            source_population_ref=source_population_ref,
            label_spec_refs=labels,
            reference_policy_ref=reference_policy_ref,
            records=ordered_records,
            reference_set_sha256=digest,
            audit=safe_audit,
        )

    @model_validator(mode="after")
    def validate_set(self) -> Self:
        _require_ref(
            self.source_population_ref,
            "external-source-population",
            "v2",
            "source_population_ref",
        )
        _require_ref(
            self.reference_policy_ref,
            "external-reference-authoring-policy",
            "v2",
            "reference_policy_ref",
        )
        if self.label_spec_refs != _sorted_refs(self.label_spec_refs):
            raise ValueError("reference-set labels must be sorted")
        for ref in self.label_spec_refs:
            _require_ref(ref, "label-spec", "v2", "label_spec_refs")
        if self.records != tuple(sorted(self.records, key=_record_key)):
            raise ValueError("reference-set records must be sorted")
        pair_keys = tuple((record.source_trace_id, record.label_spec_ref) for record in self.records)
        if len(pair_keys) != len(set(pair_keys)):
            raise ValueError("reference set contains duplicate trace-label pairs")
        if {record.label_spec_ref for record in self.records} - set(self.label_spec_refs):
            raise ValueError("reference set contains an undeclared label")
        refs = _sorted_refs(
            (
                self.source_population_ref,
                self.reference_policy_ref,
                *self.label_spec_refs,
                *(record.to_ref() for record in self.records),
            )
        )
        if self.audit.input_refs != refs:
            raise ValueError("reference-set audit refs are stale")
        digest = _carried_sha256(
            self,
            exclude={"reference_set_id", "reference_set_sha256", "audit"},
        )
        if (
            self.reference_set_sha256 != digest
            or self.reference_set_id != f"external-reference-set://sha256/{digest}"
        ):
            raise ValueError("external reference-set identity is stale")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="external-reference-set",
            object_id=self.reference_set_id,
            object_version="private-v1",
            object_sha256=self.reference_set_sha256,
        )


def _carried_sha256(
    value: ContractModelV2,
    *,
    exclude: set[str],
) -> str:
    payload = value.model_dump(
        mode="python",
        exclude={"schema_version", *exclude},
    )
    canonical = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _record_key(
    record: ExternalReferenceRecordV1,
) -> tuple[str, str, str]:
    return (
        record.source_trace_id,
        record.label_spec_ref.object_id,
        record.reference_record_id,
    )


def _sorted_refs(
    refs: tuple[ObjectRef, ...],
) -> tuple[ObjectRef, ...]:
    return tuple(
        sorted(
            set(refs),
            key=lambda ref: (
                ref.object_type,
                ref.object_id,
                ref.object_version,
                ref.object_sha256,
            ),
        )
    )


def _require_ref(
    ref: ObjectRef,
    object_type: str,
    object_version: str,
    label: str,
) -> None:
    if ref.object_type != object_type or ref.object_version != object_version:
        raise ValueError(f"{label} must reference {object_type}/{object_version}")


__all__ = [
    "ExternalReferenceAuthorKindV1",
    "ExternalReferenceRecordV1",
    "ExternalReferenceSetV1",
    "ExternalReferenceStateV1",
]
