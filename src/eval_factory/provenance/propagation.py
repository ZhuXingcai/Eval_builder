from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator

from eval_factory.contracts.core import ContractAudit, ContractModel, Identifier, ObjectRef
from eval_factory.contracts.safety import (
    ContentRiskLabel,
    Disposition,
    OriginClass,
    ProvenanceDecision,
    TaintEdge,
    TaintLabel,
    Visibility,
)
from eval_factory.provenance.decisions import ProvenanceDecisionInput, ProvenanceDecisionTable

TAINT_PROPAGATION_POLICY_VERSION: Literal["taint-propagation/r2-03-v1"] = "taint-propagation/r2-03-v1"

_DISPOSITION_RANK = {
    Disposition.ALLOW_INPUT_EVIDENCE: 0,
    Disposition.ALLOW_STRUCTURE_ONLY: 1,
    Disposition.ALLOW_EXTERNAL_LEAD_ONLY: 2,
    Disposition.NEEDS_REVIEW: 3,
    Disposition.QUARANTINE: 4,
    Disposition.REJECT: 5,
}


class DerivedContentOperation(StrEnum):
    COPY = "COPY"
    MOVE = "MOVE"
    ARCHIVE = "ARCHIVE"
    EXTRACTION = "EXTRACTION"
    FILE_WRITE = "FILE_WRITE"
    FILE_EDIT = "FILE_EDIT"
    SUMMARY = "SUMMARY"
    TRANSLATION = "TRANSLATION"
    EMBEDDING = "EMBEDDING"
    STRUCTURAL_PROJECTION = "STRUCTURAL_PROJECTION"
    UNKNOWN = "UNKNOWN"


class DerivationVerification(StrEnum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    AMBIGUOUS = "AMBIGUOUS"


class PropagationUncertainty(StrEnum):
    MISSING_PARENT = "MISSING_PARENT"
    UNKNOWN_OPERATION = "UNKNOWN_OPERATION"
    UNVERIFIED_TRANSFORM = "UNVERIFIED_TRANSFORM"
    AMBIGUOUS_TRANSFORM = "AMBIGUOUS_TRANSFORM"
    UNSCANNABLE_TARGET = "UNSCANNABLE_TARGET"


class TaintPropagationRequest(ContractModel):
    schema_version: Literal["eval-factory/taint-propagation-request/r2-03"] = (
        "eval-factory/taint-propagation-request/r2-03"
    )
    child_ref: ObjectRef
    child_origin_class: OriginClass
    visibility: Visibility
    operation: DerivedContentOperation
    parent_decisions: tuple[ProvenanceDecision, ...] = ()
    transform_verification: DerivationVerification = DerivationVerification.VERIFIED
    target_content_risk_labels: frozenset[ContentRiskLabel] = frozenset()
    extra_taint_labels: frozenset[TaintLabel] = frozenset()
    rule_id: Identifier
    source_event_refs: tuple[ObjectRef, ...] = ()
    confidence: float = Field(ge=0, le=1)
    audit: ContractAudit

    @field_validator("child_origin_class", mode="before")
    @classmethod
    def parse_child_origin_class(cls, value: object) -> OriginClass:
        if isinstance(value, OriginClass):
            return value
        if isinstance(value, str):
            return OriginClass(value)
        raise TypeError("child_origin_class must be an OriginClass")

    @field_validator("visibility", mode="before")
    @classmethod
    def parse_visibility(cls, value: object) -> Visibility:
        if isinstance(value, Visibility):
            return value
        if isinstance(value, str):
            return Visibility(value)
        raise TypeError("visibility must be a Visibility")

    @field_validator("operation", mode="before")
    @classmethod
    def parse_operation(cls, value: object) -> DerivedContentOperation:
        if isinstance(value, DerivedContentOperation):
            return value
        if isinstance(value, str):
            return DerivedContentOperation(value)
        raise TypeError("operation must be a DerivedContentOperation")

    @field_validator("transform_verification", mode="before")
    @classmethod
    def parse_transform_verification(cls, value: object) -> DerivationVerification:
        if isinstance(value, DerivationVerification):
            return value
        if isinstance(value, str):
            return DerivationVerification(value)
        raise TypeError("transform_verification must be a DerivationVerification")

    @field_validator("target_content_risk_labels", mode="before")
    @classmethod
    def parse_target_content_risk_labels(cls, value: object) -> frozenset[ContentRiskLabel]:
        if isinstance(value, frozenset):
            return frozenset(
                item if isinstance(item, ContentRiskLabel) else ContentRiskLabel(item) for item in value
            )
        if isinstance(value, (list, tuple, set)):
            return frozenset(ContentRiskLabel(item) for item in value)
        raise TypeError("target_content_risk_labels must be a collection")

    @field_validator("extra_taint_labels", mode="before")
    @classmethod
    def parse_extra_taint_labels(cls, value: object) -> frozenset[TaintLabel]:
        if isinstance(value, frozenset):
            return frozenset(item if isinstance(item, TaintLabel) else TaintLabel(item) for item in value)
        if isinstance(value, (list, tuple, set)):
            return frozenset(TaintLabel(item) for item in value)
        raise TypeError("extra_taint_labels must be a collection")


class TaintPropagationResult(ContractModel):
    schema_version: Literal["eval-factory/taint-propagation-result/r2-03"] = (
        "eval-factory/taint-propagation-result/r2-03"
    )
    taint_edge: TaintEdge
    child_decision: ProvenanceDecision
    uncertainties: tuple[PropagationUncertainty, ...] = ()
    policy_version: Literal["taint-propagation/r2-03-v1"] = TAINT_PROPAGATION_POLICY_VERSION
    audit: ContractAudit

    @field_validator("uncertainties", mode="before")
    @classmethod
    def parse_uncertainties(cls, value: object) -> tuple[PropagationUncertainty, ...]:
        if isinstance(value, tuple):
            return tuple(
                item if isinstance(item, PropagationUncertainty) else PropagationUncertainty(item)
                for item in value
            )
        if isinstance(value, (list, set, frozenset)):
            return tuple(PropagationUncertainty(item) for item in value)
        raise TypeError("uncertainties must be a collection")


class TaintPropagationEngine:
    policy_version = TAINT_PROPAGATION_POLICY_VERSION

    def propagate(self, request: TaintPropagationRequest) -> TaintPropagationResult:
        parents = _sorted_parent_decisions(request.parent_decisions)
        parent_refs = _unique_object_refs(tuple(parent.subject_ref for parent in parents))
        parent_decision_refs = _unique_object_refs(tuple(_decision_ref(parent) for parent in parents))
        source_event_refs = _unique_object_refs(
            tuple(ref for parent in parents for ref in parent.source_event_refs) + request.source_event_refs
        )
        uncertainties = _uncertainties(request)
        taint_labels = _taint_labels(request, parents, uncertainties)
        content_risk_labels = _content_risk_labels(request, parents, uncertainties)
        operation_minimum = _operation_minimum_disposition(request.operation)

        baseline = ProvenanceDecisionTable().decide(
            ProvenanceDecisionInput(
                subject_ref=request.child_ref,
                origin_class=request.child_origin_class,
                visibility=request.visibility,
                taint_labels=taint_labels,
                content_risk_labels=content_risk_labels,
                derived_from=parent_refs,
                source_event_refs=source_event_refs,
                rule_ids=(request.rule_id,),
                confidence=request.confidence,
                audit=request.audit,
            )
        )
        disposition = _most_restrictive(
            (
                baseline.disposition,
                operation_minimum,
                *(parent.disposition for parent in parents),
            )
        )
        taint_edge = _taint_edge(
            request=request,
            parent_refs=parent_refs,
            parent_decision_refs=parent_decision_refs,
            uncertainties=uncertainties,
            policy_version=self.policy_version,
        )
        child_decision = _child_decision(
            request=request,
            parent_refs=parent_refs,
            source_event_refs=source_event_refs,
            taint_labels=taint_labels,
            content_risk_labels=content_risk_labels,
            disposition=disposition,
            taint_edge=taint_edge,
            policy_version=self.policy_version,
        )
        return TaintPropagationResult(
            taint_edge=taint_edge,
            child_decision=child_decision,
            uncertainties=uncertainties,
            policy_version=self.policy_version,
            audit=request.audit,
        )


def _taint_edge(
    *,
    request: TaintPropagationRequest,
    parent_refs: tuple[ObjectRef, ...],
    parent_decision_refs: tuple[ObjectRef, ...],
    uncertainties: tuple[PropagationUncertainty, ...],
    policy_version: str,
) -> TaintEdge:
    payload = {
        "parent_refs": [item.model_dump(mode="json", exclude_none=False) for item in parent_refs],
        "parent_decision_refs": [
            item.model_dump(mode="json", exclude_none=False) for item in parent_decision_refs
        ],
        "child_ref": request.child_ref.model_dump(mode="json", exclude_none=False),
        "operation": request.operation.value,
        "rule_id": request.rule_id,
        "transform_verified": _transform_verified(request),
        "uncertainties": [item.value for item in uncertainties],
        "policy_version": policy_version,
    }
    return TaintEdge(
        taint_edge_id=_stable_id("taint-edge", payload),
        parent_refs=parent_refs,
        child_ref=request.child_ref,
        operation=request.operation.value,
        rule_id=request.rule_id,
        transform_verified=_transform_verified(request),
        audit=request.audit,
    )


def _child_decision(
    *,
    request: TaintPropagationRequest,
    parent_refs: tuple[ObjectRef, ...],
    source_event_refs: tuple[ObjectRef, ...],
    taint_labels: frozenset[TaintLabel],
    content_risk_labels: frozenset[ContentRiskLabel],
    disposition: Disposition,
    taint_edge: TaintEdge,
    policy_version: str,
) -> ProvenanceDecision:
    rule_ids = _unique_identifiers((request.rule_id, taint_edge.rule_id))
    payload = {
        "subject_ref": request.child_ref.model_dump(mode="json", exclude_none=False),
        "origin_class": request.child_origin_class.value,
        "visibility": request.visibility.value,
        "disposition": disposition.value,
        "taint_labels": sorted(item.value for item in taint_labels),
        "content_risk_labels": sorted(item.value for item in content_risk_labels),
        "derived_from": [item.model_dump(mode="json", exclude_none=False) for item in parent_refs],
        "source_event_refs": [item.model_dump(mode="json", exclude_none=False) for item in source_event_refs],
        "rule_ids": rule_ids,
        "taint_edge_id": taint_edge.taint_edge_id,
        "policy_version": policy_version,
        "subject_sha256": request.child_ref.object_sha256,
    }
    return ProvenanceDecision(
        provenance_decision_id=_stable_id("provenance-decision", payload),
        subject_ref=request.child_ref,
        origin_class=request.child_origin_class,
        taint_labels=taint_labels,
        content_risk_labels=content_risk_labels,
        visibility=request.visibility,
        disposition=disposition,
        derived_from=parent_refs,
        rule_ids=rule_ids,
        source_event_refs=source_event_refs,
        confidence=request.confidence,
        review_required=_review_required(disposition, taint_labels, content_risk_labels),
        policy_version=policy_version,
        subject_sha256=request.child_ref.object_sha256,
        audit=request.audit,
    )


def _uncertainties(request: TaintPropagationRequest) -> tuple[PropagationUncertainty, ...]:
    values: set[PropagationUncertainty] = set()
    if not request.parent_decisions:
        values.add(PropagationUncertainty.MISSING_PARENT)
    if request.operation is DerivedContentOperation.UNKNOWN:
        values.add(PropagationUncertainty.UNKNOWN_OPERATION)
    if request.transform_verification is DerivationVerification.UNVERIFIED:
        values.add(PropagationUncertainty.UNVERIFIED_TRANSFORM)
    if request.transform_verification is DerivationVerification.AMBIGUOUS:
        values.add(PropagationUncertainty.AMBIGUOUS_TRANSFORM)
    if (
        request.operation is DerivedContentOperation.UNKNOWN
        or request.transform_verification is DerivationVerification.AMBIGUOUS
        or ContentRiskLabel.UNSCANNABLE_CONTENT in request.target_content_risk_labels
    ):
        values.add(PropagationUncertainty.UNSCANNABLE_TARGET)
    return tuple(sorted(values, key=lambda item: item.value))


def _taint_labels(
    request: TaintPropagationRequest,
    parents: tuple[ProvenanceDecision, ...],
    uncertainties: tuple[PropagationUncertainty, ...],
) -> frozenset[TaintLabel]:
    values: set[TaintLabel] = set(request.extra_taint_labels)
    for parent in parents:
        values.update(parent.taint_labels)
    if any(
        item
        in {
            PropagationUncertainty.MISSING_PARENT,
            PropagationUncertainty.UNKNOWN_OPERATION,
            PropagationUncertainty.UNVERIFIED_TRANSFORM,
            PropagationUncertainty.AMBIGUOUS_TRANSFORM,
        }
        for item in uncertainties
    ):
        values.add(TaintLabel.UNKNOWN_DERIVATION)
    return frozenset(values)


def _content_risk_labels(
    request: TaintPropagationRequest,
    parents: tuple[ProvenanceDecision, ...],
    uncertainties: tuple[PropagationUncertainty, ...],
) -> frozenset[ContentRiskLabel]:
    values: set[ContentRiskLabel] = set(request.target_content_risk_labels)
    for parent in parents:
        values.update(parent.content_risk_labels)
    if PropagationUncertainty.UNSCANNABLE_TARGET in uncertainties:
        values.add(ContentRiskLabel.UNSCANNABLE_CONTENT)
    return frozenset(values)


def _operation_minimum_disposition(operation: DerivedContentOperation) -> Disposition:
    if operation in {
        DerivedContentOperation.SUMMARY,
        DerivedContentOperation.TRANSLATION,
        DerivedContentOperation.EMBEDDING,
        DerivedContentOperation.STRUCTURAL_PROJECTION,
        DerivedContentOperation.UNKNOWN,
    }:
        return Disposition.QUARANTINE
    return Disposition.ALLOW_INPUT_EVIDENCE


def _most_restrictive(values: tuple[Disposition, ...]) -> Disposition:
    return max(values, key=lambda item: _DISPOSITION_RANK[item])


def _review_required(
    disposition: Disposition,
    taint_labels: frozenset[TaintLabel],
    content_risk_labels: frozenset[ContentRiskLabel],
) -> bool:
    if disposition in {Disposition.NEEDS_REVIEW, Disposition.QUARANTINE, Disposition.REJECT}:
        return True
    return bool(taint_labels or content_risk_labels)


def _transform_verified(request: TaintPropagationRequest) -> bool:
    return request.transform_verification is DerivationVerification.VERIFIED


def _sorted_parent_decisions(
    values: tuple[ProvenanceDecision, ...],
) -> tuple[ProvenanceDecision, ...]:
    return tuple(sorted(values, key=lambda item: _object_ref_key(item.subject_ref)))


def _decision_ref(decision: ProvenanceDecision) -> ObjectRef:
    return ObjectRef(
        object_type="provenance-decision",
        object_id=decision.provenance_decision_id,
        object_version=decision.policy_version,
        object_sha256=decision.canonical_sha256(),
    )


def _unique_object_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    refs: dict[tuple[str, str, str, str], ObjectRef] = {}
    for value in values:
        refs.setdefault(_object_ref_key(value), value)
    return tuple(refs[key] for key in sorted(refs))


def _object_ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (value.object_type, value.object_id, value.object_version, value.object_sha256)


def _unique_identifiers(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _stable_id(kind: str, payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return f"{kind}://sha256/{hashlib.sha256(encoded).hexdigest()}"
