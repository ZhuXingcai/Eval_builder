from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import ContractAudit, ContractModel, Identifier, ObjectRef, TypedAttribute
from eval_factory.contracts.safety import (
    ContentRiskLabel,
    Disposition,
    ProjectionPolicy,
    ProvenanceDecision,
    TaintLabel,
    Visibility,
)
from eval_factory.provenance.decisions import ProvenanceDecisionInput, ProvenanceDecisionTable

EVIDENCE_VIEW_POLICY_VERSION: Literal["evidence-views/r2-05-v1"] = "evidence-views/r2-05-v1"
EVIDENCE_VIEW_RULE_ID: Literal["evidence-views/r2-05-v1"] = "evidence-views/r2-05-v1"
EVIDENCE_VIEW_SUBJECT_SCHEMA_VERSION: Literal["eval-factory/evidence-view-subject/r2-05"] = (
    "eval-factory/evidence-view-subject/r2-05"
)

_DISPOSITION_RANK = {
    Disposition.ALLOW_INPUT_EVIDENCE: 0,
    Disposition.ALLOW_STRUCTURE_ONLY: 1,
    Disposition.ALLOW_EXTERNAL_LEAD_ONLY: 2,
    Disposition.NEEDS_REVIEW: 3,
    Disposition.QUARANTINE: 4,
    Disposition.REJECT: 5,
}

_NON_AUDIT_DENIED_TAINTS = frozenset(
    {
        TaintLabel.FINAL_OUTPUT_DERIVED,
        TaintLabel.PRIVATE_REFERENCE_DERIVED,
        TaintLabel.GRADER_RULE_DERIVED,
        TaintLabel.UNKNOWN_DERIVATION,
        TaintLabel.UNTRUSTED_INSTRUCTION_DERIVED,
    }
)
_NON_AUDIT_DENIED_RISKS = frozenset(ContentRiskLabel)
_RAW_TRACE_OBJECT_TYPES = frozenset({"raw-trace", "raw-traj", "trace-raw"})


class EvidenceViewPolicyError(RuntimeError):
    pass


class EvidenceViewPrincipalType(StrEnum):
    PRIVILEGED_AUDITOR = "privileged-auditor"
    DEFAULT_SAFE = "default-safe"
    TASK_AUTHOR = "task-author"
    ATTACHMENT_PRODUCER = "attachment-producer"
    EVALUATOR = "evaluator"
    CONTESTANT = "contestant"


class EvidenceViewPurpose(StrEnum):
    AUDIT = "audit"
    DEFAULT_SAFE = "default-safe"
    TASK_AUTHORING = "task-authoring"
    ATTACHMENT_PRODUCTION = "attachment-production"
    EVALUATION = "evaluation"
    CONTESTANT_RUNTIME = "contestant-runtime"


class EvidenceProjectionMode(StrEnum):
    CONTENT = "CONTENT"
    STRUCTURE = "STRUCTURE"
    EXTERNAL_LEAD = "EXTERNAL_LEAD"
    AUDIT = "AUDIT"


class EvidenceProjectionExclusionReason(StrEnum):
    CHARACTER_BUDGET_EXCEEDED = "CHARACTER_BUDGET_EXCEEDED"
    CONTENT_RISK_DENIED = "CONTENT_RISK_DENIED"
    DISPOSITION_DENIED = "DISPOSITION_DENIED"
    MISSING_PROJECTION_DATA = "MISSING_PROJECTION_DATA"
    OBJECT_TYPE_DENIED = "OBJECT_TYPE_DENIED"
    RAW_TRACE_DENIED = "RAW_TRACE_DENIED"
    TAINT_DENIED = "TAINT_DENIED"
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    VISIBILITY_DENIED = "VISIBILITY_DENIED"


class EvidenceViewPrincipal(ContractModel):
    schema_version: Literal["eval-factory/evidence-view-principal/r2-05"] = (
        "eval-factory/evidence-view-principal/r2-05"
    )
    principal_id: Identifier
    principal_type: EvidenceViewPrincipalType
    allowed_purposes: frozenset[EvidenceViewPurpose] = Field(min_length=1)
    max_subjects: int = Field(ge=1)
    max_characters: int = Field(ge=0)
    audit: bool = False
    case_id: Identifier | None = None
    reason: str | None = Field(default=None, min_length=1, max_length=512)

    @field_validator("principal_type", mode="before")
    @classmethod
    def parse_principal_type(cls, value: object) -> EvidenceViewPrincipalType:
        if isinstance(value, EvidenceViewPrincipalType):
            return value
        if isinstance(value, str):
            return EvidenceViewPrincipalType(value)
        raise TypeError("principal_type must be an EvidenceViewPrincipalType")

    @field_validator("allowed_purposes", mode="before")
    @classmethod
    def parse_allowed_purposes(cls, value: object) -> frozenset[EvidenceViewPurpose]:
        if isinstance(value, frozenset):
            return frozenset(
                item if isinstance(item, EvidenceViewPurpose) else EvidenceViewPurpose(item) for item in value
            )
        if isinstance(value, (list, tuple, set)):
            return frozenset(EvidenceViewPurpose(item) for item in value)
        raise TypeError("allowed_purposes must be a collection")


class EvidenceViewSubject(ContractModel):
    schema_version: Literal["eval-factory/evidence-view-subject/r2-05"] = EVIDENCE_VIEW_SUBJECT_SCHEMA_VERSION
    subject_ref: ObjectRef
    decision: ProvenanceDecision
    projection_text: str | None = Field(default=None, max_length=1_000_000)
    structure_fields: tuple[TypedAttribute, ...] = ()
    external_uri: str | None = Field(default=None, min_length=3, max_length=2048)
    source_schema_version: Identifier = EVIDENCE_VIEW_SUBJECT_SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_decision_binding(self) -> EvidenceViewSubject:
        if self.decision.subject_ref != self.subject_ref:
            raise ValueError("evidence view subject_ref must match decision subject_ref")
        return self


class EvidenceViewRequest(ContractModel):
    schema_version: Literal["eval-factory/evidence-view-request/r2-05"] = (
        "eval-factory/evidence-view-request/r2-05"
    )
    principal: EvidenceViewPrincipal
    purpose: EvidenceViewPurpose
    subjects: tuple[EvidenceViewSubject, ...] = Field(min_length=1)
    requested_fields: tuple[str, ...] = ()
    source_event_refs: tuple[ObjectRef, ...] = ()
    max_characters: int = Field(ge=0)
    audit: ContractAudit

    @field_validator("purpose", mode="before")
    @classmethod
    def parse_purpose(cls, value: object) -> EvidenceViewPurpose:
        if isinstance(value, EvidenceViewPurpose):
            return value
        if isinstance(value, str):
            return EvidenceViewPurpose(value)
        raise TypeError("purpose must be an EvidenceViewPurpose")


class EvidenceProjectionItem(ContractModel):
    schema_version: Literal["eval-factory/evidence-projection-item/r2-05"] = (
        "eval-factory/evidence-projection-item/r2-05"
    )
    projection_item_id: Identifier
    source_ref: ObjectRef
    projected_ref: ObjectRef
    projection_mode: EvidenceProjectionMode
    content: str | None = None
    structure_fields: tuple[TypedAttribute, ...] = ()
    external_uri: str | None = Field(default=None, min_length=3, max_length=2048)
    child_decision: ProvenanceDecision

    @field_validator("projection_mode", mode="before")
    @classmethod
    def parse_projection_mode(cls, value: object) -> EvidenceProjectionMode:
        if isinstance(value, EvidenceProjectionMode):
            return value
        if isinstance(value, str):
            return EvidenceProjectionMode(value)
        raise TypeError("projection_mode must be an EvidenceProjectionMode")


class EvidenceProjectionExclusion(ContractModel):
    schema_version: Literal["eval-factory/evidence-projection-exclusion/r2-05"] = (
        "eval-factory/evidence-projection-exclusion/r2-05"
    )
    subject_ref: ObjectRef
    reason: EvidenceProjectionExclusionReason
    detail: str | None = Field(default=None, min_length=1, max_length=512)

    @field_validator("reason", mode="before")
    @classmethod
    def parse_reason(cls, value: object) -> EvidenceProjectionExclusionReason:
        if isinstance(value, EvidenceProjectionExclusionReason):
            return value
        if isinstance(value, str):
            return EvidenceProjectionExclusionReason(value)
        raise TypeError("reason must be an EvidenceProjectionExclusionReason")


class EvidenceViewResult(ContractModel):
    schema_version: Literal["eval-factory/evidence-view-result/r2-05"] = (
        "eval-factory/evidence-view-result/r2-05"
    )
    view_result_id: Identifier
    principal_id: Identifier
    principal_type: EvidenceViewPrincipalType
    purpose: EvidenceViewPurpose
    projection_policy: ProjectionPolicy
    included_items: tuple[EvidenceProjectionItem, ...] = ()
    excluded_subjects: tuple[EvidenceProjectionExclusion, ...] = ()
    returned_characters: int = Field(ge=0)
    max_characters: int = Field(ge=0)
    audit_event_ref: ObjectRef | None = None
    policy_version: Literal["evidence-views/r2-05-v1"] = EVIDENCE_VIEW_POLICY_VERSION
    audit: ContractAudit

    @field_validator("principal_type", mode="before")
    @classmethod
    def parse_principal_type(cls, value: object) -> EvidenceViewPrincipalType:
        if isinstance(value, EvidenceViewPrincipalType):
            return value
        if isinstance(value, str):
            return EvidenceViewPrincipalType(value)
        raise TypeError("principal_type must be an EvidenceViewPrincipalType")

    @field_validator("purpose", mode="before")
    @classmethod
    def parse_purpose(cls, value: object) -> EvidenceViewPurpose:
        if isinstance(value, EvidenceViewPurpose):
            return value
        if isinstance(value, str):
            return EvidenceViewPurpose(value)
        raise TypeError("purpose must be an EvidenceViewPurpose")

    @model_validator(mode="after")
    def enforce_budget(self) -> EvidenceViewResult:
        if self.returned_characters > self.max_characters:
            raise ValueError("returned characters exceed view budget")
        return self


@dataclass(frozen=True)
class _BuiltinViewPolicy:
    principal_type: EvidenceViewPrincipalType
    purpose: EvidenceViewPurpose
    recursive_allow_fields: tuple[str, ...]
    denied_object_types: frozenset[str]
    allowed_visibilities: frozenset[Visibility]
    output_visibility: Visibility
    audit_required: bool = False


_COMMON_SOURCE_SCHEMA_VERSIONS = (EVIDENCE_VIEW_SUBJECT_SCHEMA_VERSION,)
_COMMON_STRUCTURE_FIELDS = (
    "structure.normalized-path",
    "structure.media-type",
    "structure.dimensions",
    "structure.field-name",
    "structure.type-signature",
)
_COMMON_DENIED_OBJECT_TYPES = frozenset(
    {
        "raw-trace",
        "raw-traj",
        "trace-raw",
        "task-draft",
        "producer-task-view",
        "rubric-set",
        "evaluator-spec",
        "reference-policy",
        "private-reference",
        "grader-rule",
    }
)

_POLICIES: dict[EvidenceViewPrincipalType, _BuiltinViewPolicy] = {
    EvidenceViewPrincipalType.PRIVILEGED_AUDITOR: _BuiltinViewPolicy(
        principal_type=EvidenceViewPrincipalType.PRIVILEGED_AUDITOR,
        purpose=EvidenceViewPurpose.AUDIT,
        recursive_allow_fields=("content", "external-uri", *_COMMON_STRUCTURE_FIELDS),
        denied_object_types=frozenset(),
        allowed_visibilities=frozenset(Visibility),
        output_visibility=Visibility.PRIVILEGED_AUDIT,
        audit_required=True,
    ),
    EvidenceViewPrincipalType.DEFAULT_SAFE: _BuiltinViewPolicy(
        principal_type=EvidenceViewPrincipalType.DEFAULT_SAFE,
        purpose=EvidenceViewPurpose.DEFAULT_SAFE,
        recursive_allow_fields=("content", "external-uri", *_COMMON_STRUCTURE_FIELDS),
        denied_object_types=_COMMON_DENIED_OBJECT_TYPES,
        allowed_visibilities=frozenset({Visibility.STAGE_PROJECTION}),
        output_visibility=Visibility.STAGE_PROJECTION,
    ),
    EvidenceViewPrincipalType.TASK_AUTHOR: _BuiltinViewPolicy(
        principal_type=EvidenceViewPrincipalType.TASK_AUTHOR,
        purpose=EvidenceViewPurpose.TASK_AUTHORING,
        recursive_allow_fields=("content", "external-uri", *_COMMON_STRUCTURE_FIELDS),
        denied_object_types=_COMMON_DENIED_OBJECT_TYPES,
        allowed_visibilities=frozenset({Visibility.STAGE_PROJECTION}),
        output_visibility=Visibility.STAGE_PROJECTION,
    ),
    EvidenceViewPrincipalType.ATTACHMENT_PRODUCER: _BuiltinViewPolicy(
        principal_type=EvidenceViewPrincipalType.ATTACHMENT_PRODUCER,
        purpose=EvidenceViewPurpose.ATTACHMENT_PRODUCTION,
        recursive_allow_fields=("content", "external-uri", *_COMMON_STRUCTURE_FIELDS),
        denied_object_types=_COMMON_DENIED_OBJECT_TYPES,
        allowed_visibilities=frozenset({Visibility.STAGE_PROJECTION}),
        output_visibility=Visibility.STAGE_PROJECTION,
    ),
    EvidenceViewPrincipalType.EVALUATOR: _BuiltinViewPolicy(
        principal_type=EvidenceViewPrincipalType.EVALUATOR,
        purpose=EvidenceViewPurpose.EVALUATION,
        recursive_allow_fields=("content", "external-uri", *_COMMON_STRUCTURE_FIELDS),
        denied_object_types=frozenset({"raw-trace", "raw-traj", "trace-raw", "producer-task-view"}),
        allowed_visibilities=frozenset({Visibility.EVALUATOR_PROJECTION}),
        output_visibility=Visibility.EVALUATOR_PROJECTION,
    ),
    EvidenceViewPrincipalType.CONTESTANT: _BuiltinViewPolicy(
        principal_type=EvidenceViewPrincipalType.CONTESTANT,
        purpose=EvidenceViewPurpose.CONTESTANT_RUNTIME,
        recursive_allow_fields=("content", "external-uri", "structure.normalized-path"),
        denied_object_types=_COMMON_DENIED_OBJECT_TYPES,
        allowed_visibilities=frozenset({Visibility.CONTESTANT_VISIBLE}),
        output_visibility=Visibility.CONTESTANT_VISIBLE,
    ),
}


class EvidenceViewEngine:
    policy_version = EVIDENCE_VIEW_POLICY_VERSION

    def project(self, request: EvidenceViewRequest) -> EvidenceViewResult:
        policy = _policy_for(request)
        selected_fields = _selected_fields(request, policy)
        projection_policy = _projection_policy(policy, selected_fields, self.policy_version)
        returned_characters = 0
        included: list[EvidenceProjectionItem] = []
        excluded: list[EvidenceProjectionExclusion] = []
        source_event_refs = _source_event_refs(request)

        for subject in request.subjects:
            _validate_source_schema(subject, policy)
            exclusion = _exclusion(subject, policy, selected_fields)
            if exclusion is not None:
                excluded.append(exclusion)
                continue
            item = _projection_item(
                subject=subject,
                policy=policy,
                selected_fields=selected_fields,
                source_event_refs=source_event_refs,
                request=request,
                policy_version=self.policy_version,
            )
            if item is None:
                excluded.append(
                    EvidenceProjectionExclusion(
                        subject_ref=subject.subject_ref,
                        reason=EvidenceProjectionExclusionReason.MISSING_PROJECTION_DATA,
                    )
                )
                continue
            item_characters = _returned_characters(item)
            if returned_characters + item_characters > request.max_characters:
                excluded.append(
                    EvidenceProjectionExclusion(
                        subject_ref=subject.subject_ref,
                        reason=EvidenceProjectionExclusionReason.CHARACTER_BUDGET_EXCEEDED,
                    )
                )
                continue
            included.append(item)
            returned_characters += item_characters

        audit_event_ref = _audit_event_ref(request, policy) if policy.audit_required else None
        result_payload = {
            "principal_id": request.principal.principal_id,
            "principal_type": request.principal.principal_type.value,
            "purpose": request.purpose.value,
            "projection_policy_id": projection_policy.projection_policy_id,
            "included_item_ids": [item.projection_item_id for item in included],
            "excluded": [
                {
                    "subject_ref": item.subject_ref.model_dump(mode="json", exclude_none=False),
                    "reason": item.reason.value,
                }
                for item in excluded
            ],
            "returned_characters": returned_characters,
            "max_characters": request.max_characters,
            "audit_event_ref": None
            if audit_event_ref is None
            else audit_event_ref.model_dump(mode="json", exclude_none=False),
            "policy_version": self.policy_version,
        }
        return EvidenceViewResult(
            view_result_id=_stable_id("evidence-view-result", result_payload),
            principal_id=request.principal.principal_id,
            principal_type=request.principal.principal_type,
            purpose=request.purpose,
            projection_policy=projection_policy,
            included_items=tuple(included),
            excluded_subjects=tuple(excluded),
            returned_characters=returned_characters,
            max_characters=request.max_characters,
            audit_event_ref=audit_event_ref,
            policy_version=self.policy_version,
            audit=request.audit,
        )


def _policy_for(request: EvidenceViewRequest) -> _BuiltinViewPolicy:
    policy = _POLICIES[request.principal.principal_type]
    if request.purpose not in request.principal.allowed_purposes:
        raise EvidenceViewPolicyError("purpose is not authorized for principal")
    if request.purpose is not policy.purpose:
        raise EvidenceViewPolicyError(
            f"purpose {request.purpose.value} is not valid for principal {policy.principal_type.value}"
        )
    if len(request.subjects) > request.principal.max_subjects:
        raise EvidenceViewPolicyError("requested subject count exceeds principal limit")
    if request.max_characters > request.principal.max_characters:
        raise EvidenceViewPolicyError("requested character budget exceeds principal limit")
    if policy.audit_required and (
        not request.principal.audit or not request.principal.case_id or not request.principal.reason
    ):
        raise EvidenceViewPolicyError("privileged audit requires audit principal, case, and reason")
    return policy


def _selected_fields(request: EvidenceViewRequest, policy: _BuiltinViewPolicy) -> frozenset[str]:
    allowed = frozenset(policy.recursive_allow_fields)
    if not request.requested_fields:
        return allowed
    requested = frozenset(request.requested_fields)
    unknown = requested - allowed
    if unknown:
        raise EvidenceViewPolicyError(
            f"requested field is not allowed by projection policy: {sorted(unknown)[0]}"
        )
    return requested


def _validate_source_schema(subject: EvidenceViewSubject, policy: _BuiltinViewPolicy) -> None:
    if subject.source_schema_version not in _COMMON_SOURCE_SCHEMA_VERSIONS:
        raise EvidenceViewPolicyError(
            f"source schema is not allowed by projection policy: {subject.source_schema_version}"
        )


def _projection_policy(
    policy: _BuiltinViewPolicy,
    selected_fields: frozenset[str],
    policy_version: str,
) -> ProjectionPolicy:
    payload = {
        "principal_type": policy.principal_type.value,
        "purpose": policy.purpose.value,
        "recursive_allow_fields": sorted(selected_fields),
        "denied_object_types": sorted(policy.denied_object_types),
        "source_schema_versions": list(_COMMON_SOURCE_SCHEMA_VERSIONS),
        "policy_version": policy_version,
    }
    return ProjectionPolicy(
        projection_policy_id=_stable_id("projection-policy", payload),
        principal_type=policy.principal_type.value,
        purpose=policy.purpose.value,
        recursive_allow_fields=tuple(sorted(selected_fields)),
        denied_object_types=tuple(sorted(policy.denied_object_types)),
        source_schema_versions=_COMMON_SOURCE_SCHEMA_VERSIONS,
        policy_version=policy_version,
    )


def _exclusion(
    subject: EvidenceViewSubject,
    policy: _BuiltinViewPolicy,
    selected_fields: frozenset[str],
) -> EvidenceProjectionExclusion | None:
    if policy.audit_required:
        return None
    if subject.subject_ref.object_type in _RAW_TRACE_OBJECT_TYPES:
        return _exclude(subject, EvidenceProjectionExclusionReason.RAW_TRACE_DENIED)
    if subject.subject_ref.object_type in policy.denied_object_types:
        return _exclude(subject, EvidenceProjectionExclusionReason.OBJECT_TYPE_DENIED)
    if subject.decision.visibility not in policy.allowed_visibilities:
        return _exclude(subject, EvidenceProjectionExclusionReason.VISIBILITY_DENIED)
    if subject.decision.taint_labels & _NON_AUDIT_DENIED_TAINTS:
        return _exclude(subject, EvidenceProjectionExclusionReason.TAINT_DENIED)
    if subject.decision.content_risk_labels & _NON_AUDIT_DENIED_RISKS:
        return _exclude(subject, EvidenceProjectionExclusionReason.CONTENT_RISK_DENIED)
    if subject.decision.disposition in {
        Disposition.NEEDS_REVIEW,
        Disposition.QUARANTINE,
        Disposition.REJECT,
    }:
        return _exclude(subject, EvidenceProjectionExclusionReason.DISPOSITION_DENIED)
    if subject.decision.disposition is Disposition.ALLOW_STRUCTURE_ONLY:
        unknown_fields = _structure_field_paths(subject) - selected_fields
        if unknown_fields:
            return _exclude(
                subject, EvidenceProjectionExclusionReason.UNKNOWN_FIELD, sorted(unknown_fields)[0]
            )
    return None


def _exclude(
    subject: EvidenceViewSubject,
    reason: EvidenceProjectionExclusionReason,
    detail: str | None = None,
) -> EvidenceProjectionExclusion:
    return EvidenceProjectionExclusion(subject_ref=subject.subject_ref, reason=reason, detail=detail)


def _projection_item(
    *,
    subject: EvidenceViewSubject,
    policy: _BuiltinViewPolicy,
    selected_fields: frozenset[str],
    source_event_refs: tuple[ObjectRef, ...],
    request: EvidenceViewRequest,
    policy_version: str,
) -> EvidenceProjectionItem | None:
    mode = _projection_mode(subject, policy)
    content: str | None = None
    structure_fields: tuple[TypedAttribute, ...] = ()
    external_uri: str | None = None
    if mode in {EvidenceProjectionMode.CONTENT, EvidenceProjectionMode.AUDIT}:
        if "content" not in selected_fields or subject.projection_text is None:
            return None
        content = subject.projection_text
    elif mode is EvidenceProjectionMode.STRUCTURE:
        fields = tuple(
            field for field in subject.structure_fields if f"structure.{field.key}" in selected_fields
        )
        if not fields:
            return None
        structure_fields = fields
    elif mode is EvidenceProjectionMode.EXTERNAL_LEAD:
        if "external-uri" not in selected_fields or subject.external_uri is None:
            return None
        external_uri = subject.external_uri

    projected_ref = _projected_ref(
        source_ref=subject.subject_ref,
        mode=mode,
        content=content,
        structure_fields=structure_fields,
        external_uri=external_uri,
        policy_version=policy_version,
    )
    child_decision = _child_decision(
        subject=subject,
        projected_ref=projected_ref,
        mode=mode,
        policy=policy,
        source_event_refs=source_event_refs,
        request=request,
        policy_version=policy_version,
    )
    payload = {
        "source_ref": subject.subject_ref.model_dump(mode="json", exclude_none=False),
        "projected_ref": projected_ref.model_dump(mode="json", exclude_none=False),
        "mode": mode.value,
        "child_decision_id": child_decision.provenance_decision_id,
        "policy_version": policy_version,
    }
    return EvidenceProjectionItem(
        projection_item_id=_stable_id("evidence-projection-item", payload),
        source_ref=subject.subject_ref,
        projected_ref=projected_ref,
        projection_mode=mode,
        content=content,
        structure_fields=structure_fields,
        external_uri=external_uri,
        child_decision=child_decision,
    )


def _projection_mode(subject: EvidenceViewSubject, policy: _BuiltinViewPolicy) -> EvidenceProjectionMode:
    if policy.audit_required:
        return EvidenceProjectionMode.AUDIT
    if subject.decision.disposition is Disposition.ALLOW_STRUCTURE_ONLY:
        return EvidenceProjectionMode.STRUCTURE
    if subject.decision.disposition is Disposition.ALLOW_EXTERNAL_LEAD_ONLY:
        return EvidenceProjectionMode.EXTERNAL_LEAD
    return EvidenceProjectionMode.CONTENT


def _projected_ref(
    *,
    source_ref: ObjectRef,
    mode: EvidenceProjectionMode,
    content: str | None,
    structure_fields: tuple[TypedAttribute, ...],
    external_uri: str | None,
    policy_version: str,
) -> ObjectRef:
    payload = {
        "source_ref": source_ref.model_dump(mode="json", exclude_none=False),
        "mode": mode.value,
        "content": content,
        "structure_fields": [
            field.model_dump(mode="json", exclude_none=False)
            for field in sorted(structure_fields, key=lambda item: item.key)
        ],
        "external_uri": external_uri,
        "policy_version": policy_version,
    }
    digest = _stable_hash(payload)
    return ObjectRef(
        object_type=f"{source_ref.object_type}-projection",
        object_id=f"evidence-view://sha256/{digest}",
        object_version=policy_version,
        object_sha256=digest,
    )


def _child_decision(
    *,
    subject: EvidenceViewSubject,
    projected_ref: ObjectRef,
    mode: EvidenceProjectionMode,
    policy: _BuiltinViewPolicy,
    source_event_refs: tuple[ObjectRef, ...],
    request: EvidenceViewRequest,
    policy_version: str,
) -> ProvenanceDecision:
    parent = subject.decision
    taints = parent.taint_labels
    risks = parent.content_risk_labels
    baseline = ProvenanceDecisionTable().decide(
        ProvenanceDecisionInput(
            subject_ref=projected_ref,
            origin_class=parent.origin_class,
            visibility=policy.output_visibility,
            taint_labels=taints,
            content_risk_labels=risks,
            derived_from=(subject.subject_ref,),
            source_event_refs=source_event_refs,
            rule_ids=(EVIDENCE_VIEW_RULE_ID,),
            confidence=parent.confidence,
            audit=request.audit,
        )
    )
    disposition = _most_restrictive(
        (
            baseline.disposition,
            _mode_minimum_disposition(mode),
            parent.disposition,
        )
    )
    rule_ids = _unique_identifiers((EVIDENCE_VIEW_RULE_ID,))
    payload = {
        "subject_ref": projected_ref.model_dump(mode="json", exclude_none=False),
        "origin_class": parent.origin_class.value,
        "visibility": policy.output_visibility.value,
        "disposition": disposition.value,
        "taint_labels": sorted(item.value for item in taints),
        "content_risk_labels": sorted(item.value for item in risks),
        "derived_from": [subject.subject_ref.model_dump(mode="json", exclude_none=False)],
        "source_event_refs": [item.model_dump(mode="json", exclude_none=False) for item in source_event_refs],
        "rule_ids": rule_ids,
        "projection_mode": mode.value,
        "policy_version": policy_version,
        "subject_sha256": projected_ref.object_sha256,
    }
    return ProvenanceDecision(
        provenance_decision_id=_stable_id("provenance-decision", payload),
        subject_ref=projected_ref,
        origin_class=parent.origin_class,
        taint_labels=taints,
        content_risk_labels=risks,
        visibility=policy.output_visibility,
        disposition=disposition,
        derived_from=(subject.subject_ref,),
        rule_ids=rule_ids,
        source_event_refs=source_event_refs,
        confidence=parent.confidence,
        review_required=_review_required(disposition, taints, risks),
        policy_version=policy_version,
        subject_sha256=projected_ref.object_sha256,
        audit=request.audit,
    )


def _mode_minimum_disposition(mode: EvidenceProjectionMode) -> Disposition:
    if mode is EvidenceProjectionMode.STRUCTURE:
        return Disposition.ALLOW_STRUCTURE_ONLY
    if mode is EvidenceProjectionMode.EXTERNAL_LEAD:
        return Disposition.ALLOW_EXTERNAL_LEAD_ONLY
    return Disposition.ALLOW_INPUT_EVIDENCE


def _source_event_refs(request: EvidenceViewRequest) -> tuple[ObjectRef, ...]:
    return _unique_object_refs(
        tuple(ref for subject in request.subjects for ref in subject.decision.source_event_refs)
        + request.source_event_refs
    )


def _audit_event_ref(request: EvidenceViewRequest, policy: _BuiltinViewPolicy) -> ObjectRef:
    payload = {
        "principal_id": request.principal.principal_id,
        "principal_type": request.principal.principal_type.value,
        "purpose": request.purpose.value,
        "case_id": request.principal.case_id,
        "reason": request.principal.reason,
        "subject_refs": [
            subject.subject_ref.model_dump(mode="json", exclude_none=False) for subject in request.subjects
        ],
        "policy_version": EVIDENCE_VIEW_POLICY_VERSION,
    }
    digest = _stable_hash(payload)
    return ObjectRef(
        object_type="evidence-view-audit-event",
        object_id=f"evidence-view-audit-event://sha256/{digest}",
        object_version=policy.output_visibility.value,
        object_sha256=digest,
    )


def _structure_field_paths(subject: EvidenceViewSubject) -> frozenset[str]:
    return frozenset(f"structure.{field.key}" for field in subject.structure_fields)


def _returned_characters(item: EvidenceProjectionItem) -> int:
    total = len(item.content or "")
    total += len(item.external_uri or "")
    for field in item.structure_fields:
        if isinstance(field.value, str):
            total += len(field.value)
    return total


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


def _unique_object_refs(values: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    refs: dict[tuple[str, str, str, str], ObjectRef] = {}
    for value in values:
        refs.setdefault(_object_ref_key(value), value)
    return tuple(refs[key] for key in sorted(refs))


def _object_ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (value.object_type, value.object_id, value.object_version, value.object_sha256)


def _unique_identifiers(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def validate_evidence_view_result_identity(result: EvidenceViewResult) -> None:
    payload = {
        "principal_id": result.principal_id,
        "principal_type": result.principal_type.value,
        "purpose": result.purpose.value,
        "projection_policy_id": result.projection_policy.projection_policy_id,
        "included_item_ids": [item.projection_item_id for item in result.included_items],
        "excluded": [
            {
                "subject_ref": item.subject_ref.model_dump(
                    mode="json",
                    exclude_none=False,
                ),
                "reason": item.reason.value,
            }
            for item in result.excluded_subjects
        ],
        "returned_characters": result.returned_characters,
        "max_characters": result.max_characters,
        "audit_event_ref": (
            None
            if result.audit_event_ref is None
            else result.audit_event_ref.model_dump(
                mode="json",
                exclude_none=False,
            )
        ),
        "policy_version": result.policy_version,
    }
    expected_id = _stable_id("evidence-view-result", payload)
    if result.view_result_id != expected_id:
        raise EvidenceViewPolicyError("evidence view result identity is stale or invalid")


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _stable_id(kind: str, payload: object) -> str:
    return f"{kind}://sha256/{_stable_hash(payload)}"
