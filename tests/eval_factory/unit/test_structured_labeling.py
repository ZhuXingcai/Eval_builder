from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.labeling_v2 import (
    LabelSpecV2,
    LabelUnresolvedReason,
    PredicateOperatorV2,
    StructuredPredicateV2,
)
from eval_factory.contracts.trace import (
    CapabilityStatus,
    SourceSpan,
    ToolCallRecord,
    ToolCallStatus,
    ToolFamily,
    TraceCapability,
    TraceEvent,
    TraceEventType,
)
from eval_factory.labeling import (
    STRUCTURED_LABELING_POLICY_VERSION,
    StructuredFactSet,
    StructuredLabelingPolicyError,
    StructuredPredicateCompiler,
    StructuredPredicateUncertainty,
)
from eval_factory.trace.normalization.models import object_ref_for_event, object_ref_for_span

HASH = "a" * 64
ROOT = Path(__file__).resolve().parents[3]
SOURCE_TRACE_ID = "source-trace://structured-labeling"
TRACE_IR_VERSION_ID = "trace-ir://structured-labeling/v1"


def _audit(created_at: datetime = datetime(2026, 7, 24, tzinfo=UTC)) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="structured-labeling-test",
        governing_versions=(VersionBinding(component="structured-labeling", version="r3-02"),),
    )


def _span(index: int = 0) -> SourceSpan:
    return SourceSpan(
        span_id=f"source-span://structured/{index}",
        source_uri="raw-traj://structured-labeling",
        source_trace_id=SOURCE_TRACE_ID,
        outer_record_index=0,
        field="request",
        raw_byte_start=index,
        raw_byte_end=index + 1,
        decoded_char_start=index,
        decoded_char_end=index + 1,
        approximate=False,
        raw_sha256=HASH,
    )


def _event(sequence: int, event_type: TraceEventType = TraceEventType.TOOL_CALL) -> TraceEvent:
    span = _span(sequence)
    return TraceEvent(
        event_id=f"trace-event://structured/{sequence}",
        trace_ir_version_id=TRACE_IR_VERSION_ID,
        sequence=sequence,
        event_type=event_type,
        role="tool" if event_type is TraceEventType.TOOL_RESULT else "assistant",
        content_ref=None,
        source_spans=(object_ref_for_span(span),),
    )


def _tool(
    sequence: int,
    *,
    family: ToolFamily,
    status: ToolCallStatus = ToolCallStatus.PAIRED_SUCCESS,
    error_signature: str | None = None,
    bind_events: bool = True,
) -> tuple[ToolCallRecord, tuple[TraceEvent, ...]]:
    call_event = _event(sequence, TraceEventType.TOOL_CALL)
    result_event = _event(sequence + 1, TraceEventType.TOOL_RESULT)
    record = ToolCallRecord(
        tool_call_record_id=f"tool-call-record://structured/{sequence}",
        trace_ir_version_id=TRACE_IR_VERSION_ID,
        call_id=f"call-{sequence}",
        original_name="WebSearch" if family is ToolFamily.SEARCH else "Bash",
        tool_family=family,
        arguments_ref=None,
        call_event_ref=object_ref_for_event(call_event) if bind_events else None,
        result_event_ref=object_ref_for_event(result_event) if bind_events else None,
        status=status,
        error_signature=error_signature,
        sequence=sequence,
    )
    return record, (call_event, result_event)


def _capability(
    capability: str = "tool_events",
    status: CapabilityStatus = CapabilityStatus.COMPLETE,
) -> TraceCapability:
    return TraceCapability(capability=capability, status=status)


def _trace_ref() -> ObjectRef:
    return ObjectRef(
        object_type="trace-envelope",
        object_id=TRACE_IR_VERSION_ID,
        object_version="stored-manifest/v1",
        object_sha256=HASH,
    )


def _fact_set(
    *,
    tool_call_records: tuple[ToolCallRecord, ...] = (),
    events: tuple[TraceEvent, ...] = (),
    capabilities: tuple[TraceCapability, ...] = (_capability(), _capability("call_result_pairing")),
) -> StructuredFactSet:
    spans = tuple(_span(index) for index in range(20))
    return StructuredFactSet(
        source_trace_id=SOURCE_TRACE_ID,
        trace_ir_version_id=TRACE_IR_VERSION_ID,
        trace_envelope_ref=_trace_ref(),
        capabilities=capabilities,
        source_spans=spans,
        events=events,
        tool_call_records=tool_call_records,
        file_observations=(),
        interaction_segments=(),
        audit=_audit(),
    )


def _predicate(
    predicate_id: str,
    *,
    field_path: str,
    operator: PredicateOperatorV2,
    expected_value: object,
    capability: str = "tool_events",
    fact_type: str = "tool-call-record",
    window_events: int | None = None,
) -> StructuredPredicateV2:
    return StructuredPredicateV2(
        predicate_id=predicate_id,
        fact_type=fact_type,
        field_path=field_path,
        operator=operator,
        expected_value=expected_value,  # type: ignore[arg-type]
        window_events=window_events,
        required_capability=capability,
        rule_version="structured-label/r3-02",
    )


def _label_spec(
    *,
    positive: tuple[StructuredPredicateV2, ...],
    negative: tuple[StructuredPredicateV2, ...] = (),
    prerequisites: tuple[StructuredPredicateV2, ...] = (),
) -> LabelSpecV2:
    return LabelSpecV2(
        label_spec_id="label-spec://structured-test/v2",
        label_version="v2",
        name="structured-test",
        requirement="Structured test label.",
        prerequisite_predicates=prerequisites,
        positive_predicates=positive,
        negative_predicates=negative,
        semantic_residual=None,
        decision_threshold=1.0,
        review_threshold=1.0,
        label_plan_ref=None,
        policy_version="labeling/r3-02-v1",
        label_spec_sha256=HASH,
        audit=_audit(),
    )


def _compile(label_spec: LabelSpecV2, fact_set: StructuredFactSet):
    return StructuredPredicateCompiler().compile(
        label_spec=label_spec,
        fact_set=fact_set,
        audit=_audit(),
    )


def test_search_tool_usage_matches_only_normalized_tool_records() -> None:
    search, events = _tool(2, family=ToolFamily.SEARCH)
    prompt_only = _event(0, TraceEventType.USER_TEXT)
    spec = _label_spec(
        positive=(
            _predicate(
                "predicate://search/positive",
                field_path="tool_family",
                operator=PredicateOperatorV2.EQUALS,
                expected_value="search",
            ),
        ),
        negative=(
            _predicate(
                "predicate://search/negative",
                field_path="tool_family",
                operator=PredicateOperatorV2.NOT_EXISTS,
                expected_value="search",
            ),
        ),
    )

    matched = _compile(spec, _fact_set(tool_call_records=(search,), events=(prompt_only, *events)))
    prompt_result = _compile(spec, _fact_set(events=(prompt_only,)))

    assert matched.positive_evidence
    assert matched.positive_evidence[0].subject_ref.object_type == "tool-call-record"
    assert matched.positive_evidence[0].source_spans
    assert not prompt_result.positive_evidence
    assert prompt_result.negative_evidence
    assert prompt_result.semantic_evaluation_required is False


def test_powershell_error_signature_matches_only_paired_error_signature() -> None:
    powershell, events = _tool(
        2,
        family=ToolFamily.SHELL,
        status=ToolCallStatus.PAIRED_ERROR,
        error_signature="powershell:access-denied:paired_error",
    )
    orphan, orphan_events = _tool(
        8,
        family=ToolFamily.SHELL,
        status=ToolCallStatus.ORPHAN_CALL,
        error_signature=None,
    )
    spec = _label_spec(
        positive=(
            _predicate(
                "predicate://powershell/positive",
                field_path="normalized_error_signature",
                operator=PredicateOperatorV2.ERROR_SIGNATURE,
                expected_value="powershell:*:paired_error",
                capability="call_result_pairing",
            ),
        )
    )

    matched = _compile(spec, _fact_set(tool_call_records=(powershell,), events=events))
    rejected = _compile(spec, _fact_set(tool_call_records=(orphan,), events=orphan_events))

    assert matched.positive_evidence
    assert matched.structured_capability_complete is True
    assert not rejected.positive_evidence


def test_not_error_signature_is_wildcard_aware_and_mutually_exclusive() -> None:
    powershell, events = _tool(
        2,
        family=ToolFamily.SHELL,
        status=ToolCallStatus.PAIRED_ERROR,
        error_signature="powershell:access-denied:paired_error",
    )
    bash, bash_events = _tool(
        8,
        family=ToolFamily.SHELL,
        status=ToolCallStatus.PAIRED_ERROR,
        error_signature="bash:failure:paired_error",
    )
    spec = _label_spec(
        positive=(
            _predicate(
                "predicate://powershell/positive",
                field_path="normalized_error_signature",
                operator=PredicateOperatorV2.ERROR_SIGNATURE,
                expected_value="powershell:*:paired_error",
                capability="call_result_pairing",
            ),
        ),
        negative=(
            _predicate(
                "predicate://powershell/negative/v2",
                field_path="normalized_error_signature",
                operator=PredicateOperatorV2.NOT_ERROR_SIGNATURE,
                expected_value="powershell:*:paired_error",
                capability="call_result_pairing",
            ),
        ),
    )

    matched = _compile(spec, _fact_set(tool_call_records=(powershell,), events=events))
    absent = _compile(spec, _fact_set(tool_call_records=(bash,), events=bash_events))

    assert matched.positive_evidence
    assert not matched.negative_evidence
    assert not absent.positive_evidence
    assert absent.negative_evidence
    assert absent.negative_evidence[0].capability_complete is True


def test_not_error_signature_rejects_positive_role_and_non_string_pattern() -> None:
    positive = _label_spec(
        positive=(
            _predicate(
                "predicate://powershell/invalid-positive",
                field_path="normalized_error_signature",
                operator=PredicateOperatorV2.NOT_ERROR_SIGNATURE,
                expected_value="powershell:*:paired_error",
                capability="call_result_pairing",
            ),
        ),
    )
    invalid_pattern = _label_spec(
        positive=(),
        negative=(
            _predicate(
                "predicate://powershell/invalid-pattern",
                field_path="normalized_error_signature",
                operator=PredicateOperatorV2.NOT_ERROR_SIGNATURE,
                expected_value=None,
                capability="call_result_pairing",
            ),
        ),
    )

    with pytest.raises(StructuredLabelingPolicyError, match="negative predicates"):
        _compile(positive, _fact_set())
    with pytest.raises(StructuredLabelingPolicyError, match="string expected value"):
        _compile(invalid_pattern, _fact_set())


def test_not_exists_requires_complete_capability_before_negative_evidence() -> None:
    spec = _label_spec(
        positive=(),
        negative=(
            _predicate(
                "predicate://search/not-exists",
                field_path="tool_family",
                operator=PredicateOperatorV2.NOT_EXISTS,
                expected_value="search",
            ),
        ),
    )
    complete = _compile(spec, _fact_set())
    partial = _compile(
        spec,
        _fact_set(capabilities=(_capability(status=CapabilityStatus.PARTIAL),)),
    )

    assert complete.negative_evidence
    assert complete.negative_evidence[0].capability_complete is True
    assert not partial.negative_evidence
    assert partial.structured_capability_complete is False
    assert LabelUnresolvedReason.INCOMPLETE_STRUCTURED_CAPABILITY in partial.unresolved_reasons


def test_sequence_window_detects_error_followed_by_later_task_action() -> None:
    failed, failed_events = _tool(
        2,
        family=ToolFamily.SHELL,
        status=ToolCallStatus.PAIRED_ERROR,
        error_signature="bash:failure:paired_error",
    )
    followup, followup_events = _tool(6, family=ToolFamily.SEARCH)
    spec = _label_spec(
        positive=(
            _predicate(
                "predicate://recovery/sequence",
                fact_type="interaction-segment",
                field_path="error_then_subsequent_task_action",
                operator=PredicateOperatorV2.SEQUENCE,
                expected_value=True,
                window_events=5,
            ),
        )
    )
    miss = _label_spec(
        positive=(
            _predicate(
                "predicate://recovery/window-miss",
                fact_type="interaction-segment",
                field_path="error_then_subsequent_task_action",
                operator=PredicateOperatorV2.WITHIN_WINDOW,
                expected_value=True,
                window_events=1,
            ),
        )
    )
    fact_set = _fact_set(
        tool_call_records=(failed, followup),
        events=(*failed_events, *followup_events),
    )

    matched = _compile(spec, fact_set)
    rejected = _compile(miss, fact_set)

    assert matched.positive_evidence
    assert not rejected.positive_evidence


def test_unsupported_inputs_and_malformed_regex_fail_closed() -> None:
    bad_fact = _label_spec(
        positive=(
            _predicate(
                "predicate://bad/fact",
                fact_type="raw-trace",
                field_path="tool_family",
                operator=PredicateOperatorV2.EXISTS,
                expected_value=None,
            ),
        )
    )
    bad_field = _label_spec(
        positive=(
            _predicate(
                "predicate://bad/field",
                field_path="raw_trace_text",
                operator=PredicateOperatorV2.EXISTS,
                expected_value=None,
            ),
        )
    )
    bad_regex = _label_spec(
        positive=(
            _predicate(
                "predicate://bad/regex",
                field_path="original_name",
                operator=PredicateOperatorV2.REGEX,
                expected_value="[",
            ),
        )
    )
    bad_sequence = _label_spec(
        positive=(
            _predicate(
                "predicate://bad/sequence",
                fact_type="interaction-segment",
                field_path="unsupported_relation",
                operator=PredicateOperatorV2.SEQUENCE,
                expected_value=True,
            ),
        )
    )

    for spec in (bad_fact, bad_field, bad_regex, bad_sequence):
        with pytest.raises(StructuredLabelingPolicyError):
            _compile(spec, _fact_set())


def test_missing_source_span_binding_is_uncertainty_not_fabricated_evidence() -> None:
    search, _events = _tool(2, family=ToolFamily.SEARCH, bind_events=False)
    spec = _label_spec(
        positive=(
            _predicate(
                "predicate://missing-span",
                field_path="tool_family",
                operator=PredicateOperatorV2.EQUALS,
                expected_value="search",
            ),
        )
    )

    result = _compile(spec, _fact_set(tool_call_records=(search,), events=()))

    assert not result.positive_evidence
    assert StructuredPredicateUncertainty.MISSING_SOURCE_SPAN_BINDING in result.uncertainties


def test_structured_result_ids_ignore_audit_time_and_hash_seed(tmp_path: Path) -> None:
    search, events = _tool(2, family=ToolFamily.SEARCH)
    spec = _label_spec(
        positive=(
            _predicate(
                "predicate://search/positive",
                field_path="tool_family",
                operator=PredicateOperatorV2.EQUALS,
                expected_value="search",
            ),
        )
    )
    compiler = StructuredPredicateCompiler()
    first = compiler.compile(
        label_spec=spec, fact_set=_fact_set(tool_call_records=(search,), events=events), audit=_audit()
    )
    second = compiler.compile(
        label_spec=spec,
        fact_set=_fact_set(tool_call_records=(search,), events=events),
        audit=_audit(datetime(2026, 7, 25, tzinfo=UTC)),
    )
    script = tmp_path / "check_structured_seed.py"
    script.write_text(
        f"""
from datetime import UTC, datetime
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.labeling_v2 import LabelSpecV2, PredicateOperatorV2, StructuredPredicateV2
from eval_factory.contracts.trace import CapabilityStatus, SourceSpan, ToolCallRecord, ToolCallStatus, ToolFamily, TraceCapability, TraceEvent, TraceEventType
from eval_factory.labeling import StructuredFactSet, StructuredPredicateCompiler
from eval_factory.trace.normalization.models import object_ref_for_event, object_ref_for_span

audit = ContractAudit(created_at=datetime(2026, 7, 24, tzinfo=UTC), created_by='seed-check', governing_versions=(VersionBinding(component='structured-labeling', version='r3-02'),))
span = SourceSpan(span_id='source-span://structured/2', source_uri='raw-traj://structured', source_trace_id='{SOURCE_TRACE_ID}', outer_record_index=0, field='request', raw_byte_start=0, raw_byte_end=1, decoded_char_start=0, decoded_char_end=1, approximate=False, raw_sha256='{HASH}')
event = TraceEvent(event_id='trace-event://structured/2', trace_ir_version_id='{TRACE_IR_VERSION_ID}', sequence=2, event_type=TraceEventType.TOOL_CALL, role='assistant', content_ref=None, source_spans=(object_ref_for_span(span),))
tool = ToolCallRecord(tool_call_record_id='tool-call-record://structured/2', trace_ir_version_id='{TRACE_IR_VERSION_ID}', call_id='call-2', original_name='WebSearch', tool_family=ToolFamily.SEARCH, arguments_ref=None, call_event_ref=object_ref_for_event(event), result_event_ref=None, status=ToolCallStatus.PAIRED_SUCCESS, error_signature=None, sequence=2)
predicate = StructuredPredicateV2(predicate_id='predicate://search/positive', fact_type='tool-call-record', field_path='tool_family', operator=PredicateOperatorV2.EQUALS, expected_value='search', required_capability='tool_events', rule_version='structured-label/r3-02')
spec = LabelSpecV2(label_spec_id='label-spec://structured-test/v2', label_version='v2', name='structured-test', requirement='Structured test label.', prerequisite_predicates=(), positive_predicates=(predicate,), negative_predicates=(), semantic_residual=None, decision_threshold=1.0, review_threshold=1.0, label_plan_ref=None, policy_version='labeling/r3-02-v1', label_spec_sha256='{HASH}', audit=audit)
fact_set = StructuredFactSet(source_trace_id='{SOURCE_TRACE_ID}', trace_ir_version_id='{TRACE_IR_VERSION_ID}', trace_envelope_ref=ObjectRef(object_type='trace-envelope', object_id='{TRACE_IR_VERSION_ID}', object_version='stored-manifest/v1', object_sha256='{HASH}'), capabilities=(TraceCapability(capability='tool_events', status=CapabilityStatus.COMPLETE),), source_spans=(span,), events=(event,), tool_call_records=(tool,), file_observations=(), interaction_segments=(), audit=audit)
result = StructuredPredicateCompiler().compile(label_spec=spec, fact_set=fact_set, audit=audit)
print(result.structured_label_result_id)
print(result.result_sha256)
""",
        encoding="utf-8",
    )
    env_one = {**os.environ, "PYTHONHASHSEED": "1"}
    env_two = {**os.environ, "PYTHONHASHSEED": "2"}
    run_one = subprocess.run(
        [sys.executable, str(script)], cwd=ROOT, env=env_one, check=True, text=True, capture_output=True
    )
    run_two = subprocess.run(
        [sys.executable, str(script)], cwd=ROOT, env=env_two, check=True, text=True, capture_output=True
    )

    assert first.structured_label_result_id == second.structured_label_result_id
    assert first.result_sha256 == second.result_sha256
    assert run_one.stdout == run_two.stdout


def test_structured_fact_set_rejects_raw_or_hidden_payload_fields() -> None:
    base = _fact_set().model_dump(mode="json")

    with pytest.raises(ValueError):
        StructuredFactSet.model_validate({**base, "raw_trace_text": "do not expose"})
    with pytest.raises(ValueError):
        StructuredFactSet.model_validate({**base, "content_blobs": []})
    assert STRUCTURED_LABELING_POLICY_VERSION == "structured-labeling/r3-02-v1"
