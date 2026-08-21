from __future__ import annotations

import hashlib
import json
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
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
    label_decision_ref,
)
from eval_factory.labeling.decision import LabelDecisionRoute


class BlindLabelObservationRecordV1(ContractModelV2):
    schema_version: Literal["eval-factory/blind-label-observation-record/private-v1"] = (
        "eval-factory/blind-label-observation-record/private-v1"
    )
    observation_record_id: Identifier
    source_trace_id: Identifier
    raw_sha256: Sha256
    trace_envelope_ref: ObjectRef
    label_spec_ref: ObjectRef
    observation_policy_ref: ObjectRef
    label_decision: LabelDecisionV2
    label_decision_ref: ObjectRef
    route: LabelDecisionRoute
    reference_access_denied: Literal[True] = True
    policy_version: Literal["external-evidence/r8-10-v1"] = EXTERNAL_EVIDENCE_POLICY_VERSION
    observation_record_sha256: Sha256
    audit: ContractAudit

    @classmethod
    def create(
        cls,
        *,
        source_trace_id: str,
        raw_sha256: str,
        trace_envelope_ref: ObjectRef,
        label_spec_ref: ObjectRef,
        observation_policy_ref: ObjectRef,
        label_decision: LabelDecisionV2,
        route: LabelDecisionRoute,
        audit: ContractAudit,
    ) -> BlindLabelObservationRecordV1:
        decision_ref = label_decision_ref(label_decision)
        refs = _sorted_refs(
            (
                trace_envelope_ref,
                label_spec_ref,
                observation_policy_ref,
                decision_ref,
            )
        )
        safe_audit = audit.model_copy(update={"input_refs": refs})
        value = cls.model_construct(
            observation_record_id="blind-label-observation-record://pending",
            source_trace_id=source_trace_id,
            raw_sha256=raw_sha256,
            trace_envelope_ref=trace_envelope_ref,
            label_spec_ref=label_spec_ref,
            observation_policy_ref=observation_policy_ref,
            label_decision=label_decision,
            label_decision_ref=decision_ref,
            route=route,
            reference_access_denied=True,
            observation_record_sha256="0" * 64,
            audit=safe_audit,
        )
        digest = _carried_sha256(
            value,
            exclude={
                "observation_record_id",
                "observation_record_sha256",
                "audit",
            },
        )
        return cls(
            observation_record_id=(f"blind-label-observation-record://sha256/{digest}"),
            source_trace_id=source_trace_id,
            raw_sha256=raw_sha256,
            trace_envelope_ref=trace_envelope_ref,
            label_spec_ref=label_spec_ref,
            observation_policy_ref=observation_policy_ref,
            label_decision=label_decision,
            label_decision_ref=decision_ref,
            route=route,
            reference_access_denied=True,
            observation_record_sha256=digest,
            audit=safe_audit,
        )

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        for ref, object_type, version, label in (
            (
                self.trace_envelope_ref,
                "trace-envelope",
                "v1",
                "trace_envelope_ref",
            ),
            (self.label_spec_ref, "label-spec", "v2", "label_spec_ref"),
            (
                self.observation_policy_ref,
                "external-observation-policy",
                "v2",
                "observation_policy_ref",
            ),
            (
                self.label_decision_ref,
                "label-decision",
                self.label_decision.policy_version,
                "label_decision_ref",
            ),
        ):
            _require_ref(ref, object_type, version, label)
        if (
            self.label_decision_ref != label_decision_ref(self.label_decision)
            or self.label_decision.trace_envelope_ref != self.trace_envelope_ref
            or self.label_decision.label_spec_ref != self.label_spec_ref
        ):
            raise ValueError("blind observation decision binding is stale")
        refs = _sorted_refs(
            (
                self.trace_envelope_ref,
                self.label_spec_ref,
                self.observation_policy_ref,
                self.label_decision_ref,
            )
        )
        if self.audit.input_refs != refs or any(
            ref.object_type.startswith("external-reference")
            or ref.object_type == "independent-label-annotation"
            for ref in refs
        ):
            raise ValueError("blind observation audit violates reference isolation")
        digest = _carried_sha256(
            self,
            exclude={
                "observation_record_id",
                "observation_record_sha256",
                "audit",
            },
        )
        if (
            self.observation_record_sha256 != digest
            or self.observation_record_id != f"blind-label-observation-record://sha256/{digest}"
        ):
            raise ValueError("blind observation record identity is stale")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="blind-label-observation-record",
            object_id=self.observation_record_id,
            object_version="private-v1",
            object_sha256=self.observation_record_sha256,
        )


class BlindLabelObservationSetV1(ContractModelV2):
    schema_version: Literal["eval-factory/blind-label-observation-set/private-v1"] = (
        "eval-factory/blind-label-observation-set/private-v1"
    )
    observation_set_id: Identifier
    source_population_ref: ObjectRef
    label_spec_refs: tuple[ObjectRef, ...] = Field(min_length=1, max_length=256)
    observation_policy_ref: ObjectRef
    records: tuple[BlindLabelObservationRecordV1, ...] = Field(
        min_length=1,
        max_length=30_000_000,
    )
    reference_access_denied: Literal[True] = True
    policy_version: Literal["external-evidence/r8-10-v1"] = EXTERNAL_EVIDENCE_POLICY_VERSION
    observation_set_sha256: Sha256
    audit: ContractAudit

    @classmethod
    def create(
        cls,
        *,
        source_population_ref: ObjectRef,
        label_spec_refs: tuple[ObjectRef, ...],
        observation_policy_ref: ObjectRef,
        records: tuple[BlindLabelObservationRecordV1, ...],
        audit: ContractAudit,
    ) -> BlindLabelObservationSetV1:
        labels = _sorted_refs(label_spec_refs)
        ordered_records = tuple(
            sorted(
                records,
                key=lambda record: (
                    record.source_trace_id,
                    record.label_spec_ref.object_id,
                ),
            )
        )
        refs = _sorted_refs(
            (
                source_population_ref,
                observation_policy_ref,
                *labels,
                *(record.to_ref() for record in ordered_records),
            )
        )
        safe_audit = audit.model_copy(update={"input_refs": refs})
        value = cls.model_construct(
            observation_set_id="blind-label-observation-set://pending",
            source_population_ref=source_population_ref,
            label_spec_refs=labels,
            observation_policy_ref=observation_policy_ref,
            records=ordered_records,
            reference_access_denied=True,
            observation_set_sha256="0" * 64,
            audit=safe_audit,
        )
        digest = _carried_sha256(
            value,
            exclude={
                "observation_set_id",
                "observation_set_sha256",
                "audit",
            },
        )
        return cls(
            observation_set_id=(f"blind-label-observation-set://sha256/{digest}"),
            source_population_ref=source_population_ref,
            label_spec_refs=labels,
            observation_policy_ref=observation_policy_ref,
            records=ordered_records,
            reference_access_denied=True,
            observation_set_sha256=digest,
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
            self.observation_policy_ref,
            "external-observation-policy",
            "v2",
            "observation_policy_ref",
        )
        if self.label_spec_refs != _sorted_refs(self.label_spec_refs):
            raise ValueError("blind observation labels must be sorted")
        pair_keys = tuple((record.source_trace_id, record.label_spec_ref) for record in self.records)
        if len(pair_keys) != len(set(pair_keys)):
            raise ValueError("blind observation set contains duplicate trace-label pairs")
        if {record.label_spec_ref for record in self.records} - set(self.label_spec_refs):
            raise ValueError("blind observation set contains undeclared labels")
        refs = _sorted_refs(
            (
                self.source_population_ref,
                self.observation_policy_ref,
                *self.label_spec_refs,
                *(record.to_ref() for record in self.records),
            )
        )
        if self.audit.input_refs != refs:
            raise ValueError("blind observation-set audit refs are stale")
        digest = _carried_sha256(
            self,
            exclude={
                "observation_set_id",
                "observation_set_sha256",
                "audit",
            },
        )
        if (
            self.observation_set_sha256 != digest
            or self.observation_set_id != f"blind-label-observation-set://sha256/{digest}"
        ):
            raise ValueError("blind observation-set identity is stale")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type="blind-label-observation-set",
            object_id=self.observation_set_id,
            object_version="private-v1",
            object_sha256=self.observation_set_sha256,
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
    "BlindLabelObservationRecordV1",
    "BlindLabelObservationSetV1",
]
