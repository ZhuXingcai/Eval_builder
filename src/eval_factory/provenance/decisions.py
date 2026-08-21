from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, field_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    Identifier,
    ObjectRef,
)
from eval_factory.contracts.safety import (
    NON_WAIVABLE_RISKS,
    NON_WAIVABLE_TAINTS,
    ContentRiskLabel,
    Disposition,
    OriginClass,
    ProvenanceDecision,
    TaintLabel,
    Visibility,
)

PROVENANCE_DECISION_TABLE_VERSION: Literal["provenance-decision-table/r2-01-v1"] = (
    "provenance-decision-table/r2-01-v1"
)

_ALLOWING_DISPOSITIONS = {
    Disposition.ALLOW_INPUT_EVIDENCE,
    Disposition.ALLOW_STRUCTURE_ONLY,
    Disposition.ALLOW_EXTERNAL_LEAD_ONLY,
}
_DISPOSITION_RANK = {
    Disposition.ALLOW_INPUT_EVIDENCE: 0,
    Disposition.ALLOW_STRUCTURE_ONLY: 1,
    Disposition.ALLOW_EXTERNAL_LEAD_ONLY: 2,
    Disposition.NEEDS_REVIEW: 3,
    Disposition.QUARANTINE: 4,
    Disposition.REJECT: 5,
}


class ProvenancePolicyError(RuntimeError):
    pass


class ProvenanceDecisionInput(ContractModel):
    schema_version: Literal["eval-factory/provenance-decision-input/r2-01"] = (
        "eval-factory/provenance-decision-input/r2-01"
    )
    subject_ref: ObjectRef
    origin_class: OriginClass
    visibility: Visibility
    taint_labels: frozenset[TaintLabel] = frozenset()
    content_risk_labels: frozenset[ContentRiskLabel] = frozenset()
    derived_from: tuple[ObjectRef, ...] = ()
    source_event_refs: tuple[ObjectRef, ...] = ()
    rule_ids: tuple[Identifier, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    audit: ContractAudit

    @field_validator("origin_class", mode="before")
    @classmethod
    def parse_origin_class(cls, value: object) -> OriginClass:
        if isinstance(value, OriginClass):
            return value
        if isinstance(value, str):
            return OriginClass(value)
        raise TypeError("origin_class must be an OriginClass")

    @field_validator("visibility", mode="before")
    @classmethod
    def parse_visibility(cls, value: object) -> Visibility:
        if isinstance(value, Visibility):
            return value
        if isinstance(value, str):
            return Visibility(value)
        raise TypeError("visibility must be a Visibility")

    @field_validator("taint_labels", mode="before")
    @classmethod
    def parse_taint_labels(cls, value: object) -> frozenset[TaintLabel]:
        if isinstance(value, frozenset):
            return frozenset(item if isinstance(item, TaintLabel) else TaintLabel(item) for item in value)
        if isinstance(value, (list, tuple, set)):
            return frozenset(TaintLabel(item) for item in value)
        raise TypeError("taint_labels must be a collection")

    @field_validator("content_risk_labels", mode="before")
    @classmethod
    def parse_content_risk_labels(cls, value: object) -> frozenset[ContentRiskLabel]:
        if isinstance(value, frozenset):
            return frozenset(
                item if isinstance(item, ContentRiskLabel) else ContentRiskLabel(item) for item in value
            )
        if isinstance(value, (list, tuple, set)):
            return frozenset(ContentRiskLabel(item) for item in value)
        raise TypeError("content_risk_labels must be a collection")


class ProvenanceDecisionTable:
    policy_version = PROVENANCE_DECISION_TABLE_VERSION

    def decide(
        self,
        request: ProvenanceDecisionInput,
        *,
        requested_disposition: Disposition | None = None,
    ) -> ProvenanceDecision:
        derived = self._derive_disposition(request)
        if requested_disposition is not None:
            if not _can_narrow(derived, requested_disposition):
                raise ProvenancePolicyError(
                    f"cannot satisfy requested disposition {requested_disposition.value} "
                    f"for derived disposition {derived.value}"
                )
            derived = requested_disposition
        return ProvenanceDecision(
            provenance_decision_id=_decision_id(request, derived, self.policy_version),
            subject_ref=request.subject_ref,
            origin_class=request.origin_class,
            taint_labels=request.taint_labels,
            content_risk_labels=request.content_risk_labels,
            visibility=request.visibility,
            disposition=derived,
            derived_from=request.derived_from,
            rule_ids=request.rule_ids,
            source_event_refs=request.source_event_refs,
            confidence=request.confidence,
            review_required=_review_required(request, derived),
            policy_version=self.policy_version,
            subject_sha256=request.subject_ref.object_sha256,
            audit=request.audit,
        )

    def _derive_disposition(self, request: ProvenanceDecisionInput) -> Disposition:
        candidates = [_origin_disposition(request.origin_class)]
        if request.taint_labels & NON_WAIVABLE_TAINTS or request.content_risk_labels & NON_WAIVABLE_RISKS:
            candidates.append(Disposition.REJECT)
        if TaintLabel.UNKNOWN_DERIVATION in request.taint_labels:
            candidates.append(Disposition.QUARANTINE)
        if ContentRiskLabel.UNSCANNABLE_CONTENT in request.content_risk_labels:
            candidates.append(Disposition.QUARANTINE)
        if ContentRiskLabel.PROMPT_INJECTION in request.content_risk_labels:
            candidates.append(Disposition.QUARANTINE)
        return max(candidates, key=lambda item: _DISPOSITION_RANK[item])


def _origin_disposition(origin: OriginClass) -> Disposition:
    if origin in {
        OriginClass.USER_SUPPLIED_INPUT,
        OriginClass.PREEXISTING_WORKSPACE_INPUT,
        OriginClass.SYSTEM_OR_HARNESS_CONTEXT,
    }:
        return Disposition.ALLOW_INPUT_EVIDENCE
    if origin is OriginClass.AGENT_RETRIEVED_EXTERNAL:
        return Disposition.ALLOW_EXTERNAL_LEAD_ONLY
    if origin in {
        OriginClass.AGENT_GENERATED_INTERMEDIATE,
        OriginClass.UNKNOWN,
    }:
        return Disposition.QUARANTINE
    if origin is OriginClass.AGENT_GENERATED_FINAL:
        return Disposition.REJECT
    raise ProvenancePolicyError(f"unsupported origin class: {origin}")


def _review_required(
    request: ProvenanceDecisionInput,
    disposition: Disposition,
) -> bool:
    if disposition in {Disposition.NEEDS_REVIEW, Disposition.QUARANTINE, Disposition.REJECT}:
        return True
    if disposition is Disposition.ALLOW_EXTERNAL_LEAD_ONLY:
        return False
    return bool(request.taint_labels or request.content_risk_labels)


def _can_narrow(
    derived: Disposition,
    requested: Disposition,
) -> bool:
    if derived not in _ALLOWING_DISPOSITIONS:
        return requested is derived
    return _DISPOSITION_RANK[requested] >= _DISPOSITION_RANK[derived]


def _decision_id(
    request: ProvenanceDecisionInput,
    disposition: Disposition,
    policy_version: str,
) -> str:
    payload = {
        "subject_ref": request.subject_ref.model_dump(mode="json", exclude_none=False),
        "origin_class": request.origin_class.value,
        "visibility": request.visibility.value,
        "disposition": disposition.value,
        "taint_labels": sorted(item.value for item in request.taint_labels),
        "content_risk_labels": sorted(item.value for item in request.content_risk_labels),
        "derived_from": [
            item.model_dump(mode="json", exclude_none=False)
            for item in sorted(
                request.derived_from,
                key=lambda ref: (ref.object_type, ref.object_id, ref.object_version, ref.object_sha256),
            )
        ],
        "source_event_refs": [
            item.model_dump(mode="json", exclude_none=False)
            for item in sorted(
                request.source_event_refs,
                key=lambda ref: (ref.object_type, ref.object_id, ref.object_version, ref.object_sha256),
            )
        ],
        "rule_ids": sorted(request.rule_ids),
        "policy_version": policy_version,
        "subject_sha256": request.subject_ref.object_sha256,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return f"provenance-decision://sha256/{hashlib.sha256(encoded).hexdigest()}"
