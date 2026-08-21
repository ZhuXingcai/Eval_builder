from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    EvidencePolarity,
    EvidenceRef,
    Identifier,
    ObjectRef,
    ScalarValue,
    Sha256,
    SourceSpanRef,
)
from eval_factory.contracts.labeling_v2 import (
    LabelSpecV2,
    LabelUnresolvedReason,
    PredicateOperatorV2,
    StructuredPredicateV2,
)
from eval_factory.contracts.trace import (
    CapabilityStatus,
    FileObservation,
    InteractionSegment,
    SourceSpan,
    ToolCallRecord,
    ToolCallStatus,
    TraceCapability,
    TraceEvent,
)
from eval_factory.trace.storage.models import StoredTraceIndex

STRUCTURED_LABELING_POLICY_VERSION: Literal["structured-labeling/r3-02-v1"] = "structured-labeling/r3-02-v1"


class StructuredLabelingPolicyError(RuntimeError):
    pass


class StructuredPredicateRole(StrEnum):
    PREREQUISITE = "prerequisite"
    POSITIVE = "positive"
    NEGATIVE = "negative"


class StructuredPredicateUncertainty(StrEnum):
    INCOMPLETE_CAPABILITY = "INCOMPLETE_CAPABILITY"
    MISSING_SOURCE_SPAN_BINDING = "MISSING_SOURCE_SPAN_BINDING"


class StructuredFactSet(ContractModel):
    schema_version: Literal["eval-factory/structured-fact-set/r3-02"] = (
        "eval-factory/structured-fact-set/r3-02"
    )
    source_trace_id: Identifier
    trace_ir_version_id: Identifier
    trace_envelope_ref: ObjectRef
    capabilities: tuple[TraceCapability, ...] = Field(min_length=1)
    source_spans: tuple[SourceSpan, ...] = ()
    events: tuple[TraceEvent, ...] = ()
    tool_call_records: tuple[ToolCallRecord, ...] = ()
    file_observations: tuple[FileObservation, ...] = ()
    interaction_segments: tuple[InteractionSegment, ...] = ()
    audit: ContractAudit

    @classmethod
    def from_stored_index(cls, stored: StoredTraceIndex, *, audit: ContractAudit) -> StructuredFactSet:
        return cls(
            source_trace_id=stored.manifest.source_trace_id,
            trace_ir_version_id=stored.trace_ir_version_id,
            trace_envelope_ref=ObjectRef(
                object_type="trace-envelope",
                object_id=stored.manifest.trace_ir_version_id,
                object_version="stored-manifest/v1",
                object_sha256=stored.manifest.canonical_sha256(),
            ),
            capabilities=_manifest_capabilities(stored),
            source_spans=stored.source_spans,
            events=tuple(_strip_event_content(item) for item in stored.events),
            tool_call_records=stored.tool_call_records,
            file_observations=stored.file_observations,
            interaction_segments=stored.interaction_segments,
            audit=audit,
        )

    @model_validator(mode="after")
    def validate_boundary(self) -> StructuredFactSet:
        if self.trace_envelope_ref.object_type != "trace-envelope":
            raise ValueError("structured fact set requires a trace-envelope ref")
        if any(event.content_ref is not None for event in self.events):
            raise ValueError("structured fact set must not include event content refs")
        return self


class StructuredPredicateResult(ContractModel):
    schema_version: Literal["eval-factory/structured-predicate-result/r3-02"] = (
        "eval-factory/structured-predicate-result/r3-02"
    )
    predicate_id: Identifier
    role: StructuredPredicateRole
    matched: bool
    capability: Identifier
    capability_complete: bool
    evidence: tuple[EvidenceRef, ...] = ()
    unresolved_reasons: frozenset[LabelUnresolvedReason] = frozenset()
    uncertainties: tuple[StructuredPredicateUncertainty, ...] = ()
    policy_version: Literal["structured-labeling/r3-02-v1"] = STRUCTURED_LABELING_POLICY_VERSION
    audit: ContractAudit

    @field_validator("role", mode="before")
    @classmethod
    def parse_role(cls, value: object) -> StructuredPredicateRole:
        if isinstance(value, StructuredPredicateRole):
            return value
        if isinstance(value, str):
            return StructuredPredicateRole(value)
        raise TypeError("role must be a StructuredPredicateRole")

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(cls, value: object) -> frozenset[LabelUnresolvedReason]:
        if isinstance(value, frozenset):
            return frozenset(
                item if isinstance(item, LabelUnresolvedReason) else LabelUnresolvedReason(item)
                for item in value
            )
        if isinstance(value, (list, tuple, set)):
            return frozenset(LabelUnresolvedReason(item) for item in value)
        raise TypeError("unresolved_reasons must be a collection")

    @field_validator("uncertainties", mode="before")
    @classmethod
    def parse_uncertainties(cls, value: object) -> tuple[StructuredPredicateUncertainty, ...]:
        if isinstance(value, tuple):
            return tuple(
                item
                if isinstance(item, StructuredPredicateUncertainty)
                else StructuredPredicateUncertainty(item)
                for item in value
            )
        if isinstance(value, (list, set, frozenset)):
            return tuple(StructuredPredicateUncertainty(item) for item in value)
        raise TypeError("uncertainties must be a collection")


class StructuredLabelResult(ContractModel):
    schema_version: Literal["eval-factory/structured-label-result/r3-02"] = (
        "eval-factory/structured-label-result/r3-02"
    )
    structured_label_result_id: Identifier
    label_spec_ref: ObjectRef
    trace_envelope_ref: ObjectRef
    prerequisite_results: tuple[StructuredPredicateResult, ...] = ()
    positive_results: tuple[StructuredPredicateResult, ...] = ()
    negative_results: tuple[StructuredPredicateResult, ...] = ()
    prerequisites_satisfied: bool
    positive_evidence: tuple[EvidenceRef, ...] = ()
    negative_evidence: tuple[EvidenceRef, ...] = ()
    structured_capability_complete: bool
    semantic_evaluation_required: bool
    unresolved_reasons: frozenset[LabelUnresolvedReason] = frozenset()
    uncertainties: tuple[StructuredPredicateUncertainty, ...] = ()
    policy_version: Literal["structured-labeling/r3-02-v1"] = STRUCTURED_LABELING_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(cls, value: object) -> frozenset[LabelUnresolvedReason]:
        if isinstance(value, frozenset):
            return frozenset(
                item if isinstance(item, LabelUnresolvedReason) else LabelUnresolvedReason(item)
                for item in value
            )
        if isinstance(value, (list, tuple, set)):
            return frozenset(LabelUnresolvedReason(item) for item in value)
        raise TypeError("unresolved_reasons must be a collection")

    @field_validator("uncertainties", mode="before")
    @classmethod
    def parse_uncertainties(cls, value: object) -> tuple[StructuredPredicateUncertainty, ...]:
        if isinstance(value, tuple):
            return tuple(
                item
                if isinstance(item, StructuredPredicateUncertainty)
                else StructuredPredicateUncertainty(item)
                for item in value
            )
        if isinstance(value, (list, set, frozenset)):
            return tuple(StructuredPredicateUncertainty(item) for item in value)
        raise TypeError("uncertainties must be a collection")


@dataclass(frozen=True)
class _PredicateMatch:
    subject_ref: ObjectRef
    source_spans: tuple[SourceSpanRef, ...]


class StructuredPredicateCompiler:
    policy_version = STRUCTURED_LABELING_POLICY_VERSION

    def compile(
        self,
        *,
        label_spec: LabelSpecV2,
        fact_set: StructuredFactSet,
        audit: ContractAudit,
    ) -> StructuredLabelResult:
        prerequisite_results = tuple(
            self._evaluate(predicate, StructuredPredicateRole.PREREQUISITE, fact_set, audit)
            for predicate in label_spec.prerequisite_predicates
        )
        positive_results = tuple(
            self._evaluate(predicate, StructuredPredicateRole.POSITIVE, fact_set, audit)
            for predicate in label_spec.positive_predicates
        )
        negative_results = tuple(
            self._evaluate(predicate, StructuredPredicateRole.NEGATIVE, fact_set, audit)
            for predicate in label_spec.negative_predicates
        )
        all_results = (*prerequisite_results, *positive_results, *negative_results)
        positive_evidence = _sort_evidence(
            tuple(evidence for result in positive_results for evidence in result.evidence)
        )
        negative_evidence = _sort_evidence(
            tuple(evidence for result in negative_results for evidence in result.evidence)
        )
        unresolved_reasons = frozenset(
            reason for result in all_results for reason in result.unresolved_reasons
        )
        uncertainties = _sort_uncertainties(
            tuple(item for result in all_results for item in result.uncertainties)
        )
        prerequisites_satisfied = all(result.matched for result in prerequisite_results)
        structured_capability_complete = all(result.capability_complete for result in all_results)
        semantic_evaluation_required = bool(label_spec.semantic_residual) and prerequisites_satisfied
        seed = _result_seed(
            label_spec=label_spec,
            fact_set=fact_set,
            prerequisite_results=prerequisite_results,
            positive_results=positive_results,
            negative_results=negative_results,
            prerequisites_satisfied=prerequisites_satisfied,
            structured_capability_complete=structured_capability_complete,
            semantic_evaluation_required=semantic_evaluation_required,
            unresolved_reasons=unresolved_reasons,
            uncertainties=uncertainties,
        )
        return StructuredLabelResult(
            structured_label_result_id=_stable_id("structured-label-result", seed),
            label_spec_ref=_label_spec_ref(label_spec),
            trace_envelope_ref=fact_set.trace_envelope_ref,
            prerequisite_results=_sort_results(prerequisite_results),
            positive_results=_sort_results(positive_results),
            negative_results=_sort_results(negative_results),
            prerequisites_satisfied=prerequisites_satisfied,
            positive_evidence=positive_evidence,
            negative_evidence=negative_evidence,
            structured_capability_complete=structured_capability_complete,
            semantic_evaluation_required=semantic_evaluation_required,
            unresolved_reasons=unresolved_reasons,
            uncertainties=uncertainties,
            result_sha256=_stable_hash(seed),
            audit=audit,
        )

    def _evaluate(
        self,
        predicate: StructuredPredicateV2,
        role: StructuredPredicateRole,
        fact_set: StructuredFactSet,
        audit: ContractAudit,
    ) -> StructuredPredicateResult:
        if (
            predicate.operator is PredicateOperatorV2.NOT_ERROR_SIGNATURE
            and role is not StructuredPredicateRole.NEGATIVE
        ):
            raise StructuredLabelingPolicyError("NOT_ERROR_SIGNATURE is valid only for negative predicates")
        fact_kind = _fact_kind(predicate.fact_type)
        capability_complete = _capability_complete(fact_set, predicate.required_capability)
        if not capability_complete:
            return StructuredPredicateResult(
                predicate_id=predicate.predicate_id,
                role=role,
                matched=False,
                capability=predicate.required_capability,
                capability_complete=False,
                unresolved_reasons=frozenset({LabelUnresolvedReason.INCOMPLETE_STRUCTURED_CAPABILITY}),
                uncertainties=(StructuredPredicateUncertainty.INCOMPLETE_CAPABILITY,),
                audit=audit,
            )
        matches = _matches(predicate, fact_kind, fact_set)
        polarity = (
            EvidencePolarity.NEGATIVE
            if role is StructuredPredicateRole.NEGATIVE
            else EvidencePolarity.POSITIVE
        )
        if predicate.operator in {
            PredicateOperatorV2.NOT_EXISTS,
            PredicateOperatorV2.NOT_ERROR_SIGNATURE,
        }:
            if matches:
                return StructuredPredicateResult(
                    predicate_id=predicate.predicate_id,
                    role=role,
                    matched=False,
                    capability=predicate.required_capability,
                    capability_complete=True,
                    audit=audit,
                )
            fallback_span = _fallback_span(fact_set)
            if fallback_span is None:
                return StructuredPredicateResult(
                    predicate_id=predicate.predicate_id,
                    role=role,
                    matched=True,
                    capability=predicate.required_capability,
                    capability_complete=True,
                    uncertainties=(StructuredPredicateUncertainty.MISSING_SOURCE_SPAN_BINDING,),
                    audit=audit,
                )
            negative_evidence = _evidence_ref(
                predicate=predicate,
                role=role,
                subject_ref=fact_set.trace_envelope_ref,
                source_spans=(fallback_span,),
                polarity=polarity,
                capability_complete=True,
            )
            return StructuredPredicateResult(
                predicate_id=predicate.predicate_id,
                role=role,
                matched=True,
                capability=predicate.required_capability,
                capability_complete=True,
                evidence=(negative_evidence,),
                audit=audit,
            )
        evidence_items: list[EvidenceRef] = []
        missing_span = False
        for match in matches:
            if not match.source_spans:
                missing_span = True
                continue
            evidence_items.append(
                _evidence_ref(
                    predicate=predicate,
                    role=role,
                    subject_ref=match.subject_ref,
                    source_spans=match.source_spans,
                    polarity=polarity,
                    capability_complete=True,
                )
            )
        uncertainties = (StructuredPredicateUncertainty.MISSING_SOURCE_SPAN_BINDING,) if missing_span else ()
        return StructuredPredicateResult(
            predicate_id=predicate.predicate_id,
            role=role,
            matched=bool(matches),
            capability=predicate.required_capability,
            capability_complete=True,
            evidence=_sort_evidence(tuple(evidence_items)),
            uncertainties=uncertainties,
            audit=audit,
        )


def _strip_event_content(event: TraceEvent) -> TraceEvent:
    if event.content_ref is None:
        return event
    return TraceEvent(
        event_id=event.event_id,
        trace_ir_version_id=event.trace_ir_version_id,
        sequence=event.sequence,
        event_type=event.event_type,
        role=event.role,
        content_ref=None,
        source_spans=event.source_spans,
        attributes=event.attributes,
    )


def _manifest_capabilities(stored: StoredTraceIndex) -> tuple[TraceCapability, ...]:
    del stored
    return (
        TraceCapability(capability="conversation_events", status=CapabilityStatus.COMPLETE),
        TraceCapability(capability="tool_events", status=CapabilityStatus.COMPLETE),
        TraceCapability(capability="call_result_pairing", status=CapabilityStatus.COMPLETE),
        TraceCapability(capability="file_timeline", status=CapabilityStatus.COMPLETE),
    )


def _fact_kind(value: str) -> str:
    normalized = value.replace("_", "").replace("-", "").lower()
    if normalized == "toolcallrecord":
        return "tool"
    if normalized == "fileobservation":
        return "file"
    if normalized == "interactionsegment":
        return "segment"
    raise StructuredLabelingPolicyError(f"unsupported structured fact type: {value}")


def _capability_complete(fact_set: StructuredFactSet, required: str) -> bool:
    candidates: dict[str, CapabilityStatus] = {
        str(item.capability): item.status for item in fact_set.capabilities
    }
    status = candidates.get(required)
    if status is None and "-" in required:
        status = candidates.get(required.replace("-", "_"))
    return status is CapabilityStatus.COMPLETE


def _matches(
    predicate: StructuredPredicateV2,
    fact_kind: str,
    fact_set: StructuredFactSet,
) -> tuple[_PredicateMatch, ...]:
    if predicate.operator is PredicateOperatorV2.NOT_ERROR_SIGNATURE and not isinstance(
        predicate.expected_value, str
    ):
        raise StructuredLabelingPolicyError("NOT_ERROR_SIGNATURE predicates require a string expected value")
    if predicate.operator in {PredicateOperatorV2.SEQUENCE, PredicateOperatorV2.WITHIN_WINDOW}:
        return _sequence_matches(predicate, fact_set)
    facts = _facts_for_kind(fact_kind, fact_set)
    _validate_field(predicate, fact_kind)
    regex = _compile_regex(predicate) if predicate.operator is PredicateOperatorV2.REGEX else None
    matches: list[_PredicateMatch] = []
    for fact in facts:
        if _fact_matches(predicate, fact, regex):
            matches.append(_match_for_fact(fact, fact_set))
    return tuple(matches)


def _facts_for_kind(
    fact_kind: str,
    fact_set: StructuredFactSet,
) -> tuple[ToolCallRecord | FileObservation | InteractionSegment, ...]:
    if fact_kind == "tool":
        return fact_set.tool_call_records
    if fact_kind == "file":
        return fact_set.file_observations
    if fact_kind == "segment":
        return fact_set.interaction_segments
    raise StructuredLabelingPolicyError(f"unsupported structured fact kind: {fact_kind}")


def _validate_field(predicate: StructuredPredicateV2, fact_kind: str) -> None:
    allowed = {
        "tool": {
            "tool_family",
            "status",
            "original_name",
            "error_signature",
            "normalized_error_signature",
            "sequence",
        },
        "file": {"logical_path", "operation", "completeness", "truncated", "sequence"},
        "segment": {
            "boundary_method",
            "sequence_start",
            "sequence_end",
            "error_then_subsequent_task_action",
        },
    }[fact_kind]
    if predicate.field_path not in allowed:
        raise StructuredLabelingPolicyError(f"unsupported structured field path: {predicate.field_path}")


def _compile_regex(predicate: StructuredPredicateV2) -> re.Pattern[str]:
    if not isinstance(predicate.expected_value, str):
        raise StructuredLabelingPolicyError("REGEX predicates require a string expected value")
    try:
        return re.compile(predicate.expected_value)
    except re.error as exc:
        raise StructuredLabelingPolicyError("malformed structured predicate regex") from exc


def _fact_matches(
    predicate: StructuredPredicateV2,
    fact: ToolCallRecord | FileObservation | InteractionSegment,
    regex: re.Pattern[str] | None,
) -> bool:
    if predicate.operator is PredicateOperatorV2.EXISTS:
        if predicate.expected_value is None:
            return True
        return _equals(_field_value(fact, predicate.field_path), predicate.expected_value)
    if predicate.operator is PredicateOperatorV2.NOT_EXISTS:
        if predicate.expected_value is None:
            return True
        return _equals(_field_value(fact, predicate.field_path), predicate.expected_value)
    if predicate.operator is PredicateOperatorV2.EQUALS:
        return _equals(_field_value(fact, predicate.field_path), predicate.expected_value)
    if predicate.operator is PredicateOperatorV2.CONTAINS:
        return str(predicate.expected_value) in str(_field_value(fact, predicate.field_path))
    if predicate.operator is PredicateOperatorV2.REGEX:
        assert regex is not None
        return bool(regex.search(str(_field_value(fact, predicate.field_path))))
    if predicate.operator is PredicateOperatorV2.ERROR_SIGNATURE:
        value = _field_value(fact, predicate.field_path)
        return isinstance(value, str) and fnmatch.fnmatchcase(value, str(predicate.expected_value))
    if predicate.operator is PredicateOperatorV2.NOT_ERROR_SIGNATURE:
        assert isinstance(predicate.expected_value, str)
        value = _field_value(fact, predicate.field_path)
        return isinstance(value, str) and fnmatch.fnmatchcase(value, predicate.expected_value)
    raise StructuredLabelingPolicyError(f"unsupported structured predicate operator: {predicate.operator}")


def _field_value(
    fact: ToolCallRecord | FileObservation | InteractionSegment,
    field_path: str,
) -> ScalarValue:
    if isinstance(fact, ToolCallRecord):
        if field_path == "normalized_error_signature":
            value = fact.error_signature
        else:
            value = getattr(fact, field_path)
    else:
        value = getattr(fact, field_path)
    if isinstance(value, StrEnum):
        return value.value
    return value


def _equals(value: ScalarValue, expected: ScalarValue) -> bool:
    if isinstance(value, str) and isinstance(expected, str):
        return value == expected
    return value == expected


def _sequence_matches(
    predicate: StructuredPredicateV2,
    fact_set: StructuredFactSet,
) -> tuple[_PredicateMatch, ...]:
    if predicate.fact_type.replace("_", "-").lower() not in {
        "interaction-segment",
        "interactionsegment",
    }:
        raise StructuredLabelingPolicyError("sequence predicates require interaction-segment facts")
    if predicate.field_path != "error_then_subsequent_task_action":
        raise StructuredLabelingPolicyError(f"unsupported structured sequence: {predicate.field_path}")
    if predicate.expected_value is not True:
        return ()
    window = predicate.window_events
    errors = [
        record
        for record in fact_set.tool_call_records
        if record.status is ToolCallStatus.PAIRED_ERROR or record.error_signature is not None
    ]
    actions = sorted(fact_set.tool_call_records, key=lambda item: (item.sequence, item.tool_call_record_id))
    matches: list[_PredicateMatch] = []
    for error in sorted(errors, key=lambda item: (item.sequence, item.tool_call_record_id)):
        for action in actions:
            if action.tool_call_record_id == error.tool_call_record_id:
                continue
            if action.sequence <= error.sequence:
                continue
            if window is not None and action.sequence - error.sequence > window:
                continue
            spans = _spans_for_tool(error, fact_set) + _spans_for_tool(action, fact_set)
            matches.append(
                _PredicateMatch(
                    subject_ref=_tool_ref(error),
                    source_spans=_unique_spans(spans),
                )
            )
            break
    return tuple(matches)


def _match_for_fact(
    fact: ToolCallRecord | FileObservation | InteractionSegment,
    fact_set: StructuredFactSet,
) -> _PredicateMatch:
    if isinstance(fact, ToolCallRecord):
        return _PredicateMatch(subject_ref=_tool_ref(fact), source_spans=_spans_for_tool(fact, fact_set))
    if isinstance(fact, FileObservation):
        return _PredicateMatch(
            subject_ref=_file_ref(fact), source_spans=_spans_for_event_refs(fact.source_event_refs, fact_set)
        )
    return _PredicateMatch(
        subject_ref=_segment_ref(fact), source_spans=_spans_for_event_refs(fact.member_event_refs, fact_set)
    )


def _spans_for_tool(record: ToolCallRecord, fact_set: StructuredFactSet) -> tuple[SourceSpanRef, ...]:
    refs = tuple(ref for ref in (record.call_event_ref, record.result_event_ref) if ref is not None)
    return _spans_for_event_refs(refs, fact_set)


def _spans_for_event_refs(
    refs: tuple[ObjectRef, ...],
    fact_set: StructuredFactSet,
) -> tuple[SourceSpanRef, ...]:
    events = {event.event_id: event for event in fact_set.events}
    spans = {span.span_id: span for span in fact_set.source_spans}
    result: list[SourceSpanRef] = []
    for ref in refs:
        event = events.get(ref.object_id)
        if event is None:
            continue
        for span_ref in event.source_spans:
            span = spans.get(span_ref.object_id)
            if span is None:
                continue
            result.append(_source_span_ref(span))
    return _unique_spans(tuple(result))


def _source_span_ref(span: SourceSpan) -> SourceSpanRef:
    return SourceSpanRef(
        span_id=span.span_id,
        source_trace_id=span.source_trace_id,
        raw_sha256=span.raw_sha256,
        approximate=span.approximate,
    )


def _fallback_span(fact_set: StructuredFactSet) -> SourceSpanRef | None:
    if not fact_set.source_spans:
        return None
    return _source_span_ref(sorted(fact_set.source_spans, key=lambda item: item.span_id)[0])


def _unique_spans(spans: tuple[SourceSpanRef, ...]) -> tuple[SourceSpanRef, ...]:
    by_id: dict[str, SourceSpanRef] = {}
    for span in spans:
        by_id.setdefault(span.span_id, span)
    return tuple(by_id[key] for key in sorted(by_id))


def _tool_ref(item: ToolCallRecord) -> ObjectRef:
    return ObjectRef(
        object_type="tool-call-record",
        object_id=item.tool_call_record_id,
        object_version="v1",
        object_sha256=item.canonical_sha256(),
    )


def _file_ref(item: FileObservation) -> ObjectRef:
    return ObjectRef(
        object_type="file-observation",
        object_id=item.observation_id,
        object_version="v1",
        object_sha256=item.canonical_sha256(),
    )


def _segment_ref(item: InteractionSegment) -> ObjectRef:
    return ObjectRef(
        object_type="interaction-segment",
        object_id=item.segment_id,
        object_version="deterministic-interaction-segments/v1",
        object_sha256=item.canonical_sha256(),
    )


def _label_spec_ref(label_spec: LabelSpecV2) -> ObjectRef:
    return ObjectRef(
        object_type="label-spec",
        object_id=label_spec.label_spec_id,
        object_version=label_spec.label_version,
        object_sha256=label_spec.canonical_sha256(),
    )


def _evidence_ref(
    *,
    predicate: StructuredPredicateV2,
    role: StructuredPredicateRole,
    subject_ref: ObjectRef,
    source_spans: tuple[SourceSpanRef, ...],
    polarity: EvidencePolarity,
    capability_complete: bool,
) -> EvidenceRef:
    payload = {
        "predicate_id": predicate.predicate_id,
        "role": role.value,
        "subject_ref": subject_ref.model_dump(mode="json", exclude_none=False),
        "source_spans": [span.model_dump(mode="json", exclude_none=False) for span in source_spans],
        "polarity": polarity.value,
        "capability": predicate.required_capability,
        "policy_version": STRUCTURED_LABELING_POLICY_VERSION,
    }
    return EvidenceRef(
        evidence_ref_id=_stable_id("evidence-ref", payload),
        subject_ref=subject_ref,
        source_spans=source_spans,
        polarity=polarity,
        capability=predicate.required_capability,
        capability_complete=capability_complete,
    )


def _sort_evidence(evidence: tuple[EvidenceRef, ...]) -> tuple[EvidenceRef, ...]:
    return tuple(sorted(evidence, key=lambda item: item.evidence_ref_id))


def _sort_results(
    results: tuple[StructuredPredicateResult, ...],
) -> tuple[StructuredPredicateResult, ...]:
    return tuple(sorted(results, key=lambda item: item.predicate_id))


def _sort_uncertainties(
    uncertainties: tuple[StructuredPredicateUncertainty, ...],
) -> tuple[StructuredPredicateUncertainty, ...]:
    return tuple(sorted(set(uncertainties), key=lambda item: item.value))


def _result_seed(
    *,
    label_spec: LabelSpecV2,
    fact_set: StructuredFactSet,
    prerequisite_results: tuple[StructuredPredicateResult, ...],
    positive_results: tuple[StructuredPredicateResult, ...],
    negative_results: tuple[StructuredPredicateResult, ...],
    prerequisites_satisfied: bool,
    structured_capability_complete: bool,
    semantic_evaluation_required: bool,
    unresolved_reasons: frozenset[LabelUnresolvedReason],
    uncertainties: tuple[StructuredPredicateUncertainty, ...],
) -> dict[str, object]:
    return {
        "label_spec_ref": _label_spec_ref(label_spec).model_dump(mode="json", exclude_none=False),
        "trace_envelope_ref": fact_set.trace_envelope_ref.model_dump(mode="json", exclude_none=False),
        "prerequisite_results": [_result_payload(item) for item in _sort_results(prerequisite_results)],
        "positive_results": [_result_payload(item) for item in _sort_results(positive_results)],
        "negative_results": [_result_payload(item) for item in _sort_results(negative_results)],
        "prerequisites_satisfied": prerequisites_satisfied,
        "structured_capability_complete": structured_capability_complete,
        "semantic_evaluation_required": semantic_evaluation_required,
        "unresolved_reasons": sorted(reason.value for reason in unresolved_reasons),
        "uncertainties": [item.value for item in uncertainties],
        "policy_version": STRUCTURED_LABELING_POLICY_VERSION,
    }


def _result_payload(result: StructuredPredicateResult) -> dict[str, object]:
    return {
        "predicate_id": result.predicate_id,
        "role": result.role.value,
        "matched": result.matched,
        "capability": result.capability,
        "capability_complete": result.capability_complete,
        "evidence": [
            item.model_dump(mode="json", exclude_none=False) for item in _sort_evidence(result.evidence)
        ],
        "unresolved_reasons": sorted(reason.value for reason in result.unresolved_reasons),
        "uncertainties": [item.value for item in result.uncertainties],
    }


def _stable_id(kind: str, payload: object) -> str:
    return f"{kind}://sha256/{_stable_hash(payload)}"


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
