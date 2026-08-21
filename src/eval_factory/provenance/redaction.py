from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import ContractAudit, ContractModel, Identifier, ObjectRef, Sha256
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

REDACTION_POLICY_VERSION: Literal["redaction/r2-04-v1"] = "redaction/r2-04-v1"
REDACTION_OPERATION: Literal["REDACTION"] = "REDACTION"
REDACTION_RULE_ID: Literal["redaction/r2-04-v1"] = "redaction/r2-04-v1"

_DISPOSITION_RANK = {
    Disposition.ALLOW_INPUT_EVIDENCE: 0,
    Disposition.ALLOW_STRUCTURE_ONLY: 1,
    Disposition.ALLOW_EXTERNAL_LEAD_ONLY: 2,
    Disposition.NEEDS_REVIEW: 3,
    Disposition.QUARANTINE: 4,
    Disposition.REJECT: 5,
}

_SECRET_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "builtin-secret/private-key/v1",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?"
            r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    ("builtin-secret/aws-access-key/v1", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("builtin-secret/bearer-token/v1", re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{20,}")),
    (
        "builtin-secret/assigned-secret/v1",
        re.compile(r"(?i)\b(?:api[_-]?key|secret|access[_-]?token)\b\s*[:=]\s*[\"']?[a-z0-9._~+/=-]{16,}"),
    ),
)


class RedactionPolicyError(RuntimeError):
    pass


class RedactionFindingKind(StrEnum):
    SECRET = "SECRET"
    RESTRICTED_PII = "RESTRICTED_PII"


class RedactionUncertainty(StrEnum):
    INVALID_CONFIGURED_PII_RULE = "INVALID_CONFIGURED_PII_RULE"
    POST_VALIDATION_FAILED = "POST_VALIDATION_FAILED"


class ConfiguredPiiRule(ContractModel):
    schema_version: Literal["eval-factory/configured-pii-rule/r2-04"] = (
        "eval-factory/configured-pii-rule/r2-04"
    )
    rule_id: Identifier
    pattern: str = Field(min_length=1, max_length=1000)
    replacement: str | None = Field(default=None, min_length=1, max_length=128)


class RedactionFinding(ContractModel):
    schema_version: Literal["eval-factory/redaction-finding/r2-04"] = "eval-factory/redaction-finding/r2-04"
    finding_id: Identifier
    kind: RedactionFindingKind
    rule_id: Identifier
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    replacement: str = Field(min_length=1, max_length=128)
    content_risk_label: ContentRiskLabel

    @field_validator("kind", mode="before")
    @classmethod
    def parse_kind(cls, value: object) -> RedactionFindingKind:
        if isinstance(value, RedactionFindingKind):
            return value
        if isinstance(value, str):
            return RedactionFindingKind(value)
        raise TypeError("kind must be a RedactionFindingKind")

    @field_validator("content_risk_label", mode="before")
    @classmethod
    def parse_content_risk_label(cls, value: object) -> ContentRiskLabel:
        if isinstance(value, ContentRiskLabel):
            return value
        if isinstance(value, str):
            return ContentRiskLabel(value)
        raise TypeError("content_risk_label must be a ContentRiskLabel")

    @model_validator(mode="after")
    def validate_span(self) -> RedactionFinding:
        if self.end <= self.start:
            raise ValueError("redaction finding end must be greater than start")
        return self


class RedactionRequest(ContractModel):
    schema_version: Literal["eval-factory/redaction-request/r2-04"] = "eval-factory/redaction-request/r2-04"
    subject_ref: ObjectRef
    source_text: str = Field(min_length=0, max_length=10_000_000)
    parent_decision: ProvenanceDecision
    configured_pii_rules: tuple[ConfiguredPiiRule, ...] = ()
    child_origin_class: OriginClass = OriginClass.SYSTEM_OR_HARNESS_CONTEXT
    visibility: Visibility = Visibility.STAGE_PROJECTION
    source_event_refs: tuple[ObjectRef, ...] = ()
    confidence: float = Field(default=1.0, ge=0, le=1)
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

    @model_validator(mode="after")
    def validate_subject_binding(self) -> RedactionRequest:
        if self.parent_decision.subject_ref != self.subject_ref:
            raise ValueError("redaction subject_ref must match parent decision subject_ref")
        return self


class RedactionResult(ContractModel):
    schema_version: Literal["eval-factory/redaction-result/r2-04"] = "eval-factory/redaction-result/r2-04"
    redaction_result_id: Identifier
    original_ref: ObjectRef
    redacted_ref: ObjectRef
    redacted_text: str
    redacted_sha256: Sha256
    findings: tuple[RedactionFinding, ...] = ()
    cleared_content_risk_labels: tuple[ContentRiskLabel, ...] = ()
    post_validation_passed: bool
    taint_edge: TaintEdge
    child_decision: ProvenanceDecision
    uncertainties: tuple[RedactionUncertainty, ...] = ()
    policy_version: Literal["redaction/r2-04-v1"] = REDACTION_POLICY_VERSION
    audit: ContractAudit

    @field_validator("cleared_content_risk_labels", mode="before")
    @classmethod
    def parse_cleared_content_risk_labels(cls, value: object) -> tuple[ContentRiskLabel, ...]:
        if isinstance(value, tuple):
            return tuple(
                item if isinstance(item, ContentRiskLabel) else ContentRiskLabel(item) for item in value
            )
        if isinstance(value, (list, set, frozenset)):
            return tuple(ContentRiskLabel(item) for item in value)
        raise TypeError("cleared_content_risk_labels must be a collection")

    @field_validator("uncertainties", mode="before")
    @classmethod
    def parse_uncertainties(cls, value: object) -> tuple[RedactionUncertainty, ...]:
        if isinstance(value, tuple):
            return tuple(
                item if isinstance(item, RedactionUncertainty) else RedactionUncertainty(item)
                for item in value
            )
        if isinstance(value, (list, set, frozenset)):
            return tuple(RedactionUncertainty(item) for item in value)
        raise TypeError("uncertainties must be a collection")


class RedactionEngine:
    policy_version = REDACTION_POLICY_VERSION

    def redact(self, request: RedactionRequest) -> RedactionResult:
        compiled_pii = _compile_pii_rules(request.configured_pii_rules)
        findings = _findings(request.subject_ref, request.source_text, compiled_pii, self.policy_version)
        redacted_text = _apply_redactions(request.source_text, findings)
        post_findings = _findings(request.subject_ref, redacted_text, compiled_pii, self.policy_version)
        post_validation_passed = not post_findings
        uncertainties = () if post_validation_passed else (RedactionUncertainty.POST_VALIDATION_FAILED,)
        cleared = _cleared_risks(findings=findings, post_findings=post_findings)
        redacted_ref = _redacted_ref(request.subject_ref, redacted_text)
        parent_refs = (request.subject_ref,)
        source_event_refs = _unique_object_refs(
            request.parent_decision.source_event_refs + request.source_event_refs
        )
        taint_edge = _taint_edge(
            request=request,
            redacted_ref=redacted_ref,
            findings=findings,
            post_validation_passed=post_validation_passed,
            policy_version=self.policy_version,
        )
        child_decision = _child_decision(
            request=request,
            redacted_ref=redacted_ref,
            parent_refs=parent_refs,
            source_event_refs=source_event_refs,
            taint_edge=taint_edge,
            cleared_risks=cleared,
            findings=findings,
            post_validation_passed=post_validation_passed,
            policy_version=self.policy_version,
        )
        result_payload = {
            "original_ref": request.subject_ref.model_dump(mode="json", exclude_none=False),
            "redacted_ref": redacted_ref.model_dump(mode="json", exclude_none=False),
            "redacted_sha256": redacted_ref.object_sha256,
            "finding_ids": [item.finding_id for item in findings],
            "cleared_content_risk_labels": [item.value for item in cleared],
            "post_validation_passed": post_validation_passed,
            "taint_edge_id": taint_edge.taint_edge_id,
            "child_decision_id": child_decision.provenance_decision_id,
            "policy_version": self.policy_version,
        }
        return RedactionResult(
            redaction_result_id=_stable_id("redaction-result", result_payload),
            original_ref=request.subject_ref,
            redacted_ref=redacted_ref,
            redacted_text=redacted_text,
            redacted_sha256=redacted_ref.object_sha256,
            findings=findings,
            cleared_content_risk_labels=cleared,
            post_validation_passed=post_validation_passed,
            taint_edge=taint_edge,
            child_decision=child_decision,
            uncertainties=uncertainties,
            policy_version=self.policy_version,
            audit=request.audit,
        )


def _compile_pii_rules(
    rules: tuple[ConfiguredPiiRule, ...],
) -> tuple[tuple[ConfiguredPiiRule, re.Pattern[str]], ...]:
    compiled: list[tuple[ConfiguredPiiRule, re.Pattern[str]]] = []
    for rule in rules:
        try:
            compiled.append((rule, re.compile(rule.pattern)))
        except re.error as exc:
            raise RedactionPolicyError(f"invalid configured PII rule {rule.rule_id}: {exc}") from exc
    return tuple(compiled)


def _findings(
    subject_ref: ObjectRef,
    text: str,
    configured_pii_rules: tuple[tuple[ConfiguredPiiRule, re.Pattern[str]], ...],
    policy_version: str,
) -> tuple[RedactionFinding, ...]:
    findings: list[RedactionFinding] = []
    for rule_id, pattern in _SECRET_RULES:
        findings.extend(
            _rule_findings(
                subject_ref=subject_ref,
                kind=RedactionFindingKind.SECRET,
                content_risk_label=ContentRiskLabel.SECRET,
                rule_id=rule_id,
                replacement="[REDACTED:SECRET]",
                text=text,
                pattern=pattern,
                policy_version=policy_version,
            )
        )
    for rule, pattern in configured_pii_rules:
        findings.extend(
            _rule_findings(
                subject_ref=subject_ref,
                kind=RedactionFindingKind.RESTRICTED_PII,
                content_risk_label=ContentRiskLabel.RESTRICTED_PII,
                rule_id=rule.rule_id,
                replacement=rule.replacement or f"[REDACTED:RESTRICTED_PII:{rule.rule_id}]",
                text=text,
                pattern=pattern,
                policy_version=policy_version,
            )
        )
    return tuple(sorted(findings, key=lambda item: (item.start, item.end, item.kind.value, item.rule_id)))


def _rule_findings(
    *,
    subject_ref: ObjectRef,
    kind: RedactionFindingKind,
    content_risk_label: ContentRiskLabel,
    rule_id: str,
    replacement: str,
    text: str,
    pattern: re.Pattern[str],
    policy_version: str,
) -> tuple[RedactionFinding, ...]:
    findings: list[RedactionFinding] = []
    for match in pattern.finditer(text):
        if match.end() <= match.start():
            continue
        payload = {
            "subject_ref": subject_ref.model_dump(mode="json", exclude_none=False),
            "kind": kind.value,
            "rule_id": rule_id,
            "start": match.start(),
            "end": match.end(),
            "replacement": replacement,
            "policy_version": policy_version,
        }
        findings.append(
            RedactionFinding(
                finding_id=_stable_id("redaction-finding", payload),
                kind=kind,
                rule_id=rule_id,
                start=match.start(),
                end=match.end(),
                replacement=replacement,
                content_risk_label=content_risk_label,
            )
        )
    return tuple(findings)


def _apply_redactions(text: str, findings: tuple[RedactionFinding, ...]) -> str:
    if not findings:
        return text
    spans = _merged_spans(findings)
    parts: list[str] = []
    cursor = 0
    for start, end, replacement in spans:
        parts.append(text[cursor:start])
        parts.append(replacement)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _merged_spans(findings: tuple[RedactionFinding, ...]) -> tuple[tuple[int, int, str], ...]:
    spans: list[tuple[int, int, set[str], set[RedactionFindingKind]]] = []
    for finding in sorted(findings, key=lambda item: (item.start, item.end, item.rule_id)):
        if not spans or finding.start >= spans[-1][1]:
            spans.append((finding.start, finding.end, {finding.replacement}, {finding.kind}))
            continue
        start, end, replacements, kinds = spans[-1]
        replacements.add(finding.replacement)
        kinds.add(finding.kind)
        spans[-1] = (start, max(end, finding.end), replacements, kinds)
    return tuple(
        (start, end, _span_replacement(replacements, kinds)) for start, end, replacements, kinds in spans
    )


def _span_replacement(replacements: set[str], kinds: set[RedactionFindingKind]) -> str:
    if len(replacements) == 1:
        return next(iter(replacements))
    if kinds == {RedactionFindingKind.SECRET}:
        return "[REDACTED:SECRET]"
    if kinds == {RedactionFindingKind.RESTRICTED_PII}:
        return "[REDACTED:RESTRICTED_PII]"
    return "[REDACTED:SENSITIVE]"


def _cleared_risks(
    *,
    findings: tuple[RedactionFinding, ...],
    post_findings: tuple[RedactionFinding, ...],
) -> tuple[ContentRiskLabel, ...]:
    source_labels = {finding.content_risk_label for finding in findings}
    remaining_labels = {finding.content_risk_label for finding in post_findings}
    return tuple(
        sorted(
            {
                label
                for label in {ContentRiskLabel.SECRET, ContentRiskLabel.RESTRICTED_PII}
                if label in source_labels and label not in remaining_labels
            },
            key=lambda item: item.value,
        )
    )


def _redacted_ref(subject_ref: ObjectRef, redacted_text: str) -> ObjectRef:
    digest = hashlib.sha256(redacted_text.encode()).hexdigest()
    return ObjectRef(
        object_type=subject_ref.object_type,
        object_id=f"redacted-subject://sha256/{digest}",
        object_version=REDACTION_POLICY_VERSION,
        object_sha256=digest,
    )


def _taint_edge(
    *,
    request: RedactionRequest,
    redacted_ref: ObjectRef,
    findings: tuple[RedactionFinding, ...],
    post_validation_passed: bool,
    policy_version: str,
) -> TaintEdge:
    payload = {
        "parent_ref": request.subject_ref.model_dump(mode="json", exclude_none=False),
        "child_ref": redacted_ref.model_dump(mode="json", exclude_none=False),
        "operation": REDACTION_OPERATION,
        "rule_id": REDACTION_RULE_ID,
        "finding_ids": [finding.finding_id for finding in findings],
        "transform_verified": post_validation_passed,
        "policy_version": policy_version,
    }
    return TaintEdge(
        taint_edge_id=_stable_id("taint-edge", payload),
        parent_refs=(request.subject_ref,),
        child_ref=redacted_ref,
        operation=REDACTION_OPERATION,
        rule_id=REDACTION_RULE_ID,
        transform_verified=post_validation_passed,
        audit=request.audit,
    )


def _child_decision(
    *,
    request: RedactionRequest,
    redacted_ref: ObjectRef,
    parent_refs: tuple[ObjectRef, ...],
    source_event_refs: tuple[ObjectRef, ...],
    taint_edge: TaintEdge,
    cleared_risks: tuple[ContentRiskLabel, ...],
    findings: tuple[RedactionFinding, ...],
    post_validation_passed: bool,
    policy_version: str,
) -> ProvenanceDecision:
    parent = request.parent_decision
    taints = set(parent.taint_labels)
    if findings:
        taints.add(TaintLabel.SENSITIVE_SOURCE_DERIVED)
    risks = set(parent.content_risk_labels) - set(cleared_risks)
    if not post_validation_passed:
        risks.add(ContentRiskLabel.UNSCANNABLE_CONTENT)
    baseline = ProvenanceDecisionTable().decide(
        ProvenanceDecisionInput(
            subject_ref=redacted_ref,
            origin_class=request.child_origin_class,
            visibility=request.visibility,
            taint_labels=frozenset(taints),
            content_risk_labels=frozenset(risks),
            derived_from=parent_refs,
            source_event_refs=source_event_refs,
            rule_ids=(REDACTION_RULE_ID,),
            confidence=request.confidence,
            audit=request.audit,
        )
    )
    disposition = _redaction_disposition(baseline.disposition, frozenset(risks))
    rule_ids = _unique_identifiers((REDACTION_RULE_ID, taint_edge.rule_id))
    payload = {
        "subject_ref": redacted_ref.model_dump(mode="json", exclude_none=False),
        "origin_class": request.child_origin_class.value,
        "visibility": request.visibility.value,
        "disposition": disposition.value,
        "taint_labels": sorted(item.value for item in taints),
        "content_risk_labels": sorted(item.value for item in risks),
        "derived_from": [item.model_dump(mode="json", exclude_none=False) for item in parent_refs],
        "source_event_refs": [item.model_dump(mode="json", exclude_none=False) for item in source_event_refs],
        "rule_ids": rule_ids,
        "taint_edge_id": taint_edge.taint_edge_id,
        "policy_version": policy_version,
        "subject_sha256": redacted_ref.object_sha256,
    }
    return ProvenanceDecision(
        provenance_decision_id=_stable_id("provenance-decision", payload),
        subject_ref=redacted_ref,
        origin_class=request.child_origin_class,
        taint_labels=frozenset(taints),
        content_risk_labels=frozenset(risks),
        visibility=request.visibility,
        disposition=disposition,
        derived_from=parent_refs,
        rule_ids=rule_ids,
        source_event_refs=source_event_refs,
        confidence=request.confidence,
        review_required=_review_required(disposition, frozenset(taints), frozenset(risks)),
        policy_version=policy_version,
        subject_sha256=redacted_ref.object_sha256,
        audit=request.audit,
    )


def _redaction_disposition(
    baseline: Disposition,
    risks: frozenset[ContentRiskLabel],
) -> Disposition:
    candidates = [baseline]
    if ContentRiskLabel.RESTRICTED_PII in risks:
        candidates.append(Disposition.QUARANTINE)
    return max(candidates, key=lambda item: _DISPOSITION_RANK[item])


def _review_required(
    disposition: Disposition,
    taints: frozenset[TaintLabel],
    risks: frozenset[ContentRiskLabel],
) -> bool:
    if disposition in {Disposition.NEEDS_REVIEW, Disposition.QUARANTINE, Disposition.REJECT}:
        return True
    return bool(taints or risks)


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
