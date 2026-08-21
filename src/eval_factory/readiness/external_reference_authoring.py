from __future__ import annotations

import fnmatch
import hashlib
import json

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionValueV2,
    LabelSpecV2,
    PredicateOperatorV2,
    SemanticResidualSpecV2,
    StructuredPredicateV2,
)
from eval_factory.contracts.trace import (
    CapabilityStatus,
    ToolCallRecord,
    ToolCallStatus,
    ToolFamily,
    TraceEnvelope,
)
from eval_factory.readiness.external_reference_models import (
    ExternalReferenceAuthorKindV1,
    ExternalReferenceRecordV1,
    ExternalReferenceStateV1,
)
from eval_factory.trace.indexing.models import (
    TraceIndexResult,
    object_ref_for_observation,
    object_ref_for_segment,
)
from eval_factory.trace.normalization.models import (
    object_ref_for_event,
    object_ref_for_span,
)
from eval_factory.trace.parsing.recovery import TraceCapabilityName

APPROVED_EXTERNAL_LABEL_NAMES = (
    "contextual_recovery_after_tool_error",
    "powershell_error_signature",
    "search_tool_usage",
)
EXTERNAL_STRUCTURED_REFERENCE_POLICY_VERSION = "external-structured-reference/r8-10-v1"


class ExternalReferenceAuthoringError(RuntimeError):
    pass


class ExternalStructuredReferenceBuilder:
    def __init__(
        self,
        *,
        annotation_contract_ref: ObjectRef,
        reference_policy_ref: ObjectRef,
    ) -> None:
        _require_ref(
            annotation_contract_ref,
            "annotation-contract-manifest",
            "v1",
            "annotation_contract_ref",
        )
        _require_ref(
            reference_policy_ref,
            "external-reference-authoring-policy",
            "v2",
            "reference_policy_ref",
        )
        self.annotation_contract_ref = annotation_contract_ref
        self.reference_policy_ref = reference_policy_ref

    def author(
        self,
        *,
        trace_index: TraceIndexResult,
        label_specs: tuple[LabelSpecV2, ...],
        audit: ContractAudit,
    ) -> tuple[ExternalReferenceRecordV1, ...]:
        specs = {spec.name: spec for spec in label_specs}
        if set(specs) != set(APPROVED_EXTERNAL_LABEL_NAMES):
            raise ExternalReferenceAuthoringError("external reference label portfolio is not exact")
        for spec in label_specs:
            _validate_label_spec_identity(spec)
        trace_envelope_ref = external_trace_envelope_ref(
            trace_index,
            audit=audit,
        )
        registered = trace_index.normalization_result.recovery_result.base_parse_result.registered_source
        records = (
            self._search_record(
                trace_index=trace_index,
                trace_envelope_ref=trace_envelope_ref,
                label_spec=specs["search_tool_usage"],
                audit=audit,
            ),
            self._powershell_record(
                trace_index=trace_index,
                trace_envelope_ref=trace_envelope_ref,
                label_spec=specs["powershell_error_signature"],
                audit=audit,
            ),
        )
        if any(
            record.source_trace_id != registered.source.source_trace_id
            or record.raw_sha256 != registered.source.raw_sha256
            for record in records
        ):
            raise ExternalReferenceAuthoringError("external structured reference source binding drifted")
        return tuple(
            sorted(
                records,
                key=lambda record: (
                    record.label_name,
                    record.reference_record_id,
                ),
            )
        )

    def _search_record(
        self,
        *,
        trace_index: TraceIndexResult,
        trace_envelope_ref: ObjectRef,
        label_spec: LabelSpecV2,
        audit: ContractAudit,
    ) -> ExternalReferenceRecordV1:
        matches = tuple(
            record
            for record in trace_index.normalization_result.tool_call_records
            if (record.tool_family is ToolFamily.SEARCH and record.call_event_ref is not None)
        )
        return self._structured_record(
            trace_index=trace_index,
            trace_envelope_ref=trace_envelope_ref,
            label_spec=label_spec,
            required_capability="tool_events",
            matches=matches,
            audit=audit,
        )

    def _powershell_record(
        self,
        *,
        trace_index: TraceIndexResult,
        trace_envelope_ref: ObjectRef,
        label_spec: LabelSpecV2,
        audit: ContractAudit,
    ) -> ExternalReferenceRecordV1:
        matches = tuple(
            record
            for record in trace_index.normalization_result.tool_call_records
            if (
                record.original_name.strip().casefold() == "powershell"
                and record.status is ToolCallStatus.PAIRED_ERROR
                and isinstance(record.error_signature, str)
                and fnmatch.fnmatchcase(
                    record.error_signature,
                    "powershell:*:paired_error",
                )
            )
        )
        return self._structured_record(
            trace_index=trace_index,
            trace_envelope_ref=trace_envelope_ref,
            label_spec=label_spec,
            required_capability="call_result_pairing",
            matches=matches,
            audit=audit,
        )

    def _structured_record(
        self,
        *,
        trace_index: TraceIndexResult,
        trace_envelope_ref: ObjectRef,
        label_spec: LabelSpecV2,
        required_capability: TraceCapabilityName,
        matches: tuple[ToolCallRecord, ...],
        audit: ContractAudit,
    ) -> ExternalReferenceRecordV1:
        recovery = trace_index.normalization_result.recovery_result
        registered = recovery.base_parse_result.registered_source
        capabilities = {capability.capability: capability.status for capability in recovery.capabilities}
        complete = capabilities.get(required_capability) is CapabilityStatus.COMPLETE
        if matches:
            expected = LabelDecisionValueV2.MATCH
            state = ExternalReferenceStateV1.REFERENCE_READY
            evidence_refs = tuple(_tool_record_ref(record) for record in matches)
        elif complete:
            expected = LabelDecisionValueV2.NO_MATCH
            state = ExternalReferenceStateV1.REFERENCE_READY
            evidence_refs = (trace_envelope_ref,)
        else:
            expected = LabelDecisionValueV2.ABSTAIN
            state = ExternalReferenceStateV1.ABSTAINED
            evidence_refs = ()
        return ExternalReferenceRecordV1.create(
            source_trace_id=registered.source.source_trace_id,
            raw_sha256=registered.source.raw_sha256,
            trace_envelope_ref=trace_envelope_ref,
            label_spec_ref=_label_spec_ref(label_spec),
            label_name=label_spec.name,
            annotation_contract_ref=self.annotation_contract_ref,
            reference_policy_ref=self.reference_policy_ref,
            expected_decision=expected,
            evidence_refs=evidence_refs,
            structured_capability_complete=complete,
            author_kind=(ExternalReferenceAuthorKindV1.DETERMINISTIC_SERVICE),
            reference_state=state,
            rule_version=EXTERNAL_STRUCTURED_REFERENCE_POLICY_VERSION,
            model_profile=None,
            prompt_version=None,
            audit=audit,
        )


def approved_external_label_specs(
    *,
    audit: ContractAudit,
) -> tuple[LabelSpecV2, ...]:
    search = _label_spec(
        label_spec_id="label-spec://search-tool-usage/v2",
        name="search_tool_usage",
        requirement=(
            "Match only when a normalized executed search-family tool call "
            "exists; complete absence is NO_MATCH and incomplete tool "
            "capability is ABSTAIN."
        ),
        prerequisite_predicates=(),
        positive_predicates=(
            StructuredPredicateV2(
                predicate_id=("predicate://search-tool-usage/positive/v2"),
                fact_type="tool-call-record",
                field_path="tool_family",
                operator=PredicateOperatorV2.EQUALS,
                expected_value="search",
                required_capability="tool_events",
                rule_version="structured-labeling/r3-02-v1",
            ),
        ),
        negative_predicates=(
            StructuredPredicateV2(
                predicate_id=("predicate://search-tool-usage/negative/v2"),
                fact_type="tool-call-record",
                field_path="tool_family",
                operator=PredicateOperatorV2.NOT_EXISTS,
                expected_value="search",
                required_capability="tool_events",
                rule_version="structured-labeling/r3-02-v1",
            ),
        ),
        semantic_residual=None,
        decision_threshold=1.0,
        review_threshold=1.0,
        audit=audit,
    )
    powershell = _label_spec(
        label_spec_id=("label-spec://powershell-error-signature/r8-10-v2"),
        policy_version="labeling/r8-10-v2",
        name="powershell_error_signature",
        requirement=(
            "Match only when an executed PowerShell call is explicitly paired "
            "to an error result with the approved normalized signature."
        ),
        prerequisite_predicates=(),
        positive_predicates=(
            StructuredPredicateV2(
                predicate_id=("predicate://powershell-error-signature/positive/r8-10-v2"),
                fact_type="tool-call-record",
                field_path="normalized_error_signature",
                operator=PredicateOperatorV2.ERROR_SIGNATURE,
                expected_value="powershell:*:paired_error",
                required_capability="call_result_pairing",
                rule_version="structured-labeling/r3-02-v1",
            ),
        ),
        negative_predicates=(
            StructuredPredicateV2(
                predicate_id=("predicate://powershell-error-signature/negative/r8-10-v2"),
                fact_type="tool-call-record",
                field_path="normalized_error_signature",
                operator=PredicateOperatorV2.NOT_ERROR_SIGNATURE,
                expected_value="powershell:*:paired_error",
                required_capability="call_result_pairing",
                rule_version="powershell-error-signature/r8-10-v2",
            ),
        ),
        semantic_residual=None,
        decision_threshold=1.0,
        review_threshold=1.0,
        audit=audit,
    )
    recovery = _label_spec(
        label_spec_id=("label-spec://contextual-recovery-after-tool-error/v2"),
        name="contextual_recovery_after_tool_error",
        requirement=(
            "Match when a proven tool error is followed by a materially "
            "different task-directed strategy that continues toward the same "
            "user intent."
        ),
        prerequisite_predicates=(
            StructuredPredicateV2(
                predicate_id=("predicate://contextual-recovery/prerequisite/v2"),
                fact_type="interaction-segment",
                field_path="error_then_subsequent_task_action",
                operator=PredicateOperatorV2.SEQUENCE,
                expected_value=True,
                window_events=50,
                required_capability="tool_events",
                rule_version="structured-labeling/r3-02-v1",
            ),
        ),
        positive_predicates=(),
        negative_predicates=(
            StructuredPredicateV2(
                predicate_id=("predicate://contextual-recovery/no-prerequisite/v2"),
                fact_type="interaction-segment",
                field_path="error_then_subsequent_task_action",
                operator=PredicateOperatorV2.NOT_EXISTS,
                expected_value=True,
                window_events=50,
                required_capability="tool_events",
                rule_version="structured-labeling/r3-02-v1",
            ),
        ),
        semantic_residual=SemanticResidualSpecV2(
            residual_id="semantic-residual://contextual-recovery/v2",
            question=(
                "Did the later action materially adapt strategy while continuing toward the same user intent?"
            ),
            evidence_bundle_purpose="label-contextual-recovery",
            allowed_evidence_types=(
                "assistant_text",
                "tool_call",
                "tool_result",
                "user_text",
            ),
            abstain_conditions=(
                "Intent continuity is ambiguous.",
                "Error or later action evidence is incomplete.",
                "The later action may be an identical retry.",
            ),
            model_profile="external-semantic-observer-v1",
            prompt_version=("contextual-recovery-observation/r8-10-v1"),
        ),
        decision_threshold=0.85,
        review_threshold=0.7,
        audit=audit,
    )
    return tuple(sorted((search, powershell, recovery), key=lambda spec: spec.name))


def _label_spec(
    *,
    label_spec_id: str,
    policy_version: str = "labeling/r8-10-v1",
    name: str,
    requirement: str,
    prerequisite_predicates: tuple[StructuredPredicateV2, ...],
    positive_predicates: tuple[StructuredPredicateV2, ...],
    negative_predicates: tuple[StructuredPredicateV2, ...],
    semantic_residual: SemanticResidualSpecV2 | None,
    decision_threshold: float,
    review_threshold: float,
    audit: ContractAudit,
) -> LabelSpecV2:
    value = LabelSpecV2(
        label_spec_id=label_spec_id,
        label_version="v2",
        name=name,
        requirement=requirement,
        prerequisite_predicates=prerequisite_predicates,
        positive_predicates=positive_predicates,
        negative_predicates=negative_predicates,
        semantic_residual=semantic_residual,
        decision_threshold=decision_threshold,
        review_threshold=review_threshold,
        label_plan_ref=None,
        policy_version=policy_version,
        label_spec_sha256="0" * 64,
        audit=audit,
    )
    digest = _label_spec_sha256(value)
    return value.model_copy(update={"label_spec_sha256": digest})


def _validate_label_spec_identity(spec: LabelSpecV2) -> None:
    if spec.label_spec_sha256 != _label_spec_sha256(spec):
        raise ExternalReferenceAuthoringError("external reference LabelSpec identity is stale")


def _label_spec_sha256(spec: LabelSpecV2) -> str:
    payload = spec.model_dump(
        mode="python",
        exclude={"label_spec_id", "label_spec_sha256", "audit"},
        exclude_none=False,
    )
    canonical = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _label_spec_ref(spec: LabelSpecV2) -> ObjectRef:
    return ObjectRef(
        object_type="label-spec",
        object_id=spec.label_spec_id,
        object_version=spec.label_version,
        object_sha256=spec.label_spec_sha256,
    )


def external_trace_envelope_ref(
    trace_index: TraceIndexResult,
    *,
    audit: ContractAudit,
) -> ObjectRef:
    normalization = trace_index.normalization_result
    recovery = normalization.recovery_result
    registered = recovery.base_parse_result.registered_source
    envelope = TraceEnvelope(
        source_trace_id=registered.source.source_trace_id,
        trace_ir_version_id=normalization.trace_ir_version_id,
        source_uri=registered.source.source_uri,
        raw_sha256=registered.source.raw_sha256,
        adapter_name=registered.source.adapter_name,
        adapter_version=registered.source.adapter_version,
        trace_ir_schema_version="v1",
        repair_policy_version=(recovery.base_parse_result.repair_policy_version),
        segmentation_policy_version=trace_index.segmentation_policy_version,
        parse_quality=recovery.parse_quality,
        capabilities=recovery.capabilities,
        event_refs=tuple(object_ref_for_event(event) for event in normalization.events),
        tool_call_refs=tuple(_tool_record_ref(record) for record in normalization.tool_call_records),
        file_observation_refs=tuple(
            object_ref_for_observation(observation) for observation in trace_index.file_observations
        ),
        segment_refs=tuple(object_ref_for_segment(segment) for segment in trace_index.interaction_segments),
        repair_map_refs=tuple(
            ObjectRef(
                object_type="repair-map",
                object_id=repair_map.repair_map_id,
                object_version="v1",
                object_sha256=repair_map.canonical_sha256(),
            )
            for repair_map in recovery.recovery_maps
        ),
        unrecoverable_span_refs=tuple(object_ref_for_span(span) for span in recovery.unrecoverable_spans),
        audit=audit,
    )
    return ObjectRef(
        object_type="trace-envelope",
        object_id=envelope.trace_ir_version_id,
        object_version="v1",
        object_sha256=_trace_envelope_behavior_sha256(envelope),
    )


def _trace_envelope_behavior_sha256(
    envelope: TraceEnvelope,
) -> str:
    payload = envelope.model_dump(
        mode="python",
        exclude={"schema_version", "audit"},
    )
    canonical = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _tool_record_ref(record: ToolCallRecord) -> ObjectRef:
    return ObjectRef(
        object_type="tool-call-record",
        object_id=record.tool_call_record_id,
        object_version="v1",
        object_sha256=record.canonical_sha256(),
    )


def _require_ref(
    ref: ObjectRef,
    object_type: str,
    object_version: str,
    label: str,
) -> None:
    if ref.object_type != object_type or ref.object_version != object_version:
        raise ExternalReferenceAuthoringError(f"{label} must reference {object_type}/{object_version}")


__all__ = [
    "APPROVED_EXTERNAL_LABEL_NAMES",
    "EXTERNAL_STRUCTURED_REFERENCE_POLICY_VERSION",
    "ExternalReferenceAuthoringError",
    "ExternalStructuredReferenceBuilder",
    "approved_external_label_specs",
    "external_trace_envelope_ref",
]
