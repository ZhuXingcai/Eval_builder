from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.labeling_v2 import (
    LabelSpecV2,
    LabelUnresolvedReason,
    PredicateOperatorV2,
    SemanticResidualSpecV2,
    StructuredPredicateV2,
)
from eval_factory.contracts.safety import EvidenceBundle
from eval_factory.contracts.trace import CapabilityStatus, TraceCapability
from eval_factory.labeling import (
    FakeSemanticResidualFixture,
    FakeSemanticResidualRunner,
    SemanticResidualOutcome,
    SemanticResidualPolicyError,
    SemanticResidualRequest,
    SemanticResidualRequestBuilder,
    SemanticResidualResult,
    StructuredFactSet,
    StructuredPredicateCompiler,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
ROOT = Path(__file__).resolve().parents[3]
SOURCE_TRACE_ID = "source-trace://semantic-residual"
TRACE_IR_VERSION_ID = "trace-ir://semantic-residual/v1"


def _audit(created_at: datetime = datetime(2026, 7, 24, tzinfo=UTC)) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="semantic-residual-test",
        governing_versions=(VersionBinding(component="semantic-residual", version="r3-03"),),
    )


def _ref(object_type: str, object_id: str, digest: str = HASH) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version="v1",
        object_sha256=digest,
    )


def _span(span_id: str = "source-span://semantic/safe") -> SourceSpanRef:
    return SourceSpanRef(span_id=span_id, source_trace_id=SOURCE_TRACE_ID, raw_sha256=HASH)


def _trace_ref() -> ObjectRef:
    return ObjectRef(
        object_type="trace-envelope",
        object_id=TRACE_IR_VERSION_ID,
        object_version="stored-manifest/v1",
        object_sha256=HASH,
    )


def _semantic_residual() -> SemanticResidualSpecV2:
    return SemanticResidualSpecV2(
        residual_id="semantic-residual://contextual-recovery/v2",
        question=(
            "Did the later action materially adapt strategy toward the same user intent using only "
            "authorized bounded evidence?"
        ),
        evidence_bundle_purpose="label-contextual-recovery",
        allowed_evidence_types=("user-text", "tool-call", "tool-result", "assistant-text"),
        abstain_conditions=("intent continuity is ambiguous", "relevant action content is unavailable"),
        model_profile="internal-semantic-labeler-v1",
        prompt_version="contextual-recovery-label-prompt/v1",
    )


def _fallback_predicate() -> StructuredPredicateV2:
    return StructuredPredicateV2(
        predicate_id="predicate://semantic/fallback",
        fact_type="tool-call-record",
        field_path="tool_family",
        operator=PredicateOperatorV2.EXISTS,
        expected_value=None,
        required_capability="tool_events",
        rule_version="semantic-residual-test/r3-03",
    )


def _label_spec(*, semantic: bool = True) -> LabelSpecV2:
    return LabelSpecV2(
        label_spec_id="label-spec://semantic-residual-test/v2",
        label_version="v2",
        name="semantic-residual-test",
        requirement="Semantic residual test label.",
        prerequisite_predicates=(),
        positive_predicates=() if semantic else (_fallback_predicate(),),
        negative_predicates=(),
        semantic_residual=_semantic_residual() if semantic else None,
        decision_threshold=0.85,
        review_threshold=0.70,
        label_plan_ref=None,
        policy_version="labeling/r3-03-v1",
        label_spec_sha256=HASH,
        audit=_audit(),
    )


def _structured_result(label_spec: LabelSpecV2 | None = None):
    spec = label_spec or _label_spec()
    fact_set = StructuredFactSet(
        source_trace_id=SOURCE_TRACE_ID,
        trace_ir_version_id=TRACE_IR_VERSION_ID,
        trace_envelope_ref=_trace_ref(),
        capabilities=(TraceCapability(capability="tool_events", status=CapabilityStatus.COMPLETE),),
        audit=_audit(),
    )
    return StructuredPredicateCompiler().compile(label_spec=spec, fact_set=fact_set, audit=_audit())


def _evidence_ref(
    evidence_id: str = "evidence-ref://semantic/safe",
    *,
    subject_ref: ObjectRef | None = None,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=evidence_id,
        subject_ref=subject_ref or _ref("file-version-projection", "evidence-view://semantic/safe"),
        source_spans=(_span(),),
        polarity=EvidencePolarity.POSITIVE,
        capability="semantic-evidence",
        capability_complete=True,
    )


def _bundle(
    *,
    evidence: tuple[EvidenceRef, ...] | None = None,
    purpose: str = "label-contextual-recovery",
    returned_characters: int = 120,
    max_characters: int = 1000,
) -> EvidenceBundle:
    return EvidenceBundle(
        evidence_bundle_id="evidence-bundle://semantic/safe",
        source_trace_id=SOURCE_TRACE_ID,
        trace_ir_version_id=TRACE_IR_VERSION_ID,
        consumer_stage="semantic-labeler",
        purpose=purpose,
        projection_policy_ref=_ref("projection-policy", "projection-policy://semantic/safe"),
        evidence=(_evidence_ref(),) if evidence is None else evidence,
        excluded_subject_refs=(),
        returned_characters=returned_characters,
        max_characters=max_characters,
        tainted_content_included=False,
        bundle_sha256=OTHER_HASH,
        audit=_audit(),
    )


def _request(
    *,
    label_spec: LabelSpecV2 | None = None,
    evidence_bundle: EvidenceBundle | None = None,
    audit: ContractAudit | None = None,
) -> SemanticResidualRequest:
    spec = label_spec or _label_spec()
    return SemanticResidualRequestBuilder().build(
        label_spec=spec,
        structured_result=_structured_result(spec),
        evidence_bundle=evidence_bundle or _bundle(),
        audit=audit or _audit(),
    )


def _fixture(
    outcome: SemanticResidualOutcome,
    *,
    request: SemanticResidualRequest,
    confidence: float = 0.91,
    selected_evidence_ref_ids: tuple[str, ...] | None = None,
    unresolved_reasons: frozenset[LabelUnresolvedReason] = frozenset(),
    model_available: bool = True,
):
    return FakeSemanticResidualFixture(
        fixture_id=f"semantic-fixture://{outcome.value.lower()}",
        outcome=outcome,
        confidence=confidence,
        selected_evidence_ref_ids=(
            (request.evidence_refs[0].evidence_ref_id,)
            if selected_evidence_ref_ids is None
            else selected_evidence_ref_ids
        ),
        unresolved_reasons=unresolved_reasons,
        explanation_code="semantic-fixture-result",
        model_available=model_available,
    )


def test_request_builder_accepts_safe_bounded_residual_evidence() -> None:
    spec = _label_spec()
    request = _request(label_spec=spec)

    assert request.label_spec_ref.object_type == "label-spec"
    assert request.trace_envelope_ref == _trace_ref()
    assert request.residual_id == spec.semantic_residual.residual_id  # type: ignore[union-attr]
    assert request.question == spec.semantic_residual.question  # type: ignore[union-attr]
    assert request.evidence_bundle_ref.object_type == "evidence-bundle"
    assert request.projection_policy_ref.object_type == "projection-policy"
    assert request.evidence_refs == _bundle().evidence
    assert request.model_profile == "internal-semantic-labeler-v1"
    assert request.prompt_version == "contextual-recovery-label-prompt/v1"
    assert request.returned_characters == 120
    assert request.max_characters == 1000
    serialized = request.model_dump_json()
    assert "raw_trace" not in serialized
    assert "grader-rule" not in serialized
    assert "final-output" not in serialized


def test_request_builder_rejects_missing_residual_failed_prerequisites_and_not_required() -> None:
    valid_spec = _label_spec()
    valid_structured = _structured_result(valid_spec)

    with pytest.raises(SemanticResidualPolicyError, match="semantic residual"):
        SemanticResidualRequestBuilder().build(
            label_spec=_label_spec(semantic=False),
            structured_result=valid_structured,
            evidence_bundle=_bundle(),
            audit=_audit(),
        )

    with pytest.raises(SemanticResidualPolicyError, match="prerequisites"):
        SemanticResidualRequestBuilder().build(
            label_spec=valid_spec,
            structured_result=valid_structured.model_copy(update={"prerequisites_satisfied": False}),
            evidence_bundle=_bundle(),
            audit=_audit(),
        )

    with pytest.raises(SemanticResidualPolicyError, match="not required"):
        SemanticResidualRequestBuilder().build(
            label_spec=valid_spec,
            structured_result=valid_structured.model_copy(update={"semantic_evaluation_required": False}),
            evidence_bundle=_bundle(),
            audit=_audit(),
        )


@pytest.mark.parametrize(
    "object_type",
    (
        "raw-trace",
        "private-reference",
        "quarantine",
        "final-output",
        "grader-rule",
        "hidden-pass-condition",
        "hidden-selection-signal",
        "answer-bearing",
    ),
)
def test_request_builder_rejects_unsafe_evidence_boundaries(object_type: str) -> None:
    unsafe_ref = _ref(object_type, f"{object_type}://semantic/unsafe")
    unsafe_bundle = _bundle(
        evidence=(_evidence_ref("evidence-ref://semantic/unsafe", subject_ref=unsafe_ref),)
    )

    with pytest.raises(SemanticResidualPolicyError, match="unsafe semantic evidence"):
        _request(evidence_bundle=unsafe_bundle)


def test_request_builder_rejects_tainted_over_budget_or_empty_bundle() -> None:
    with pytest.raises(SemanticResidualPolicyError, match="tainted"):
        _request(evidence_bundle=_bundle().model_copy(update={"tainted_content_included": True}))

    with pytest.raises(SemanticResidualPolicyError, match="budget"):
        _request(evidence_bundle=_bundle().model_copy(update={"returned_characters": 1001}))

    with pytest.raises(SemanticResidualPolicyError, match="safe evidence"):
        _request(evidence_bundle=_bundle(evidence=()))

    with pytest.raises(SemanticResidualPolicyError, match="purpose"):
        _request(evidence_bundle=_bundle(purpose="wrong-purpose"))


def test_fake_runner_returns_match_and_no_match_with_selected_semantic_evidence() -> None:
    request = _request()
    match = FakeSemanticResidualRunner().run(
        request,
        fixture=_fixture(SemanticResidualOutcome.MATCH, request=request),
        audit=_audit(),
    )
    no_match = FakeSemanticResidualRunner().run(
        request,
        fixture=_fixture(SemanticResidualOutcome.NO_MATCH, request=request, confidence=0.2),
        audit=_audit(),
    )

    assert match.outcome is SemanticResidualOutcome.MATCH
    assert match.semantic_evidence == request.evidence_refs
    assert match.confidence == 0.91
    assert match.model_profile == request.model_profile
    assert match.prompt_version == request.prompt_version
    assert not match.unresolved_reasons
    assert no_match.outcome is SemanticResidualOutcome.NO_MATCH
    assert no_match.semantic_evidence == request.evidence_refs
    assert no_match.confidence == 0.2


def test_fake_runner_returns_abstain_for_ambiguous_or_low_confidence_evidence() -> None:
    request = _request()

    result = FakeSemanticResidualRunner().run(
        request,
        fixture=_fixture(
            SemanticResidualOutcome.ABSTAIN,
            request=request,
            confidence=0.3,
            selected_evidence_ref_ids=(),
            unresolved_reasons=frozenset(
                {
                    LabelUnresolvedReason.AMBIGUOUS_EVIDENCE,
                    LabelUnresolvedReason.LOW_CONFIDENCE,
                }
            ),
        ),
        audit=_audit(),
    )

    assert result.outcome is SemanticResidualOutcome.ABSTAIN
    assert result.semantic_evidence == ()
    assert result.unresolved_reasons == frozenset(
        {LabelUnresolvedReason.AMBIGUOUS_EVIDENCE, LabelUnresolvedReason.LOW_CONFIDENCE}
    )


def test_fake_runner_returns_model_unavailable_without_fabricated_evidence() -> None:
    request = _request()

    result = FakeSemanticResidualRunner().run(
        request,
        fixture=_fixture(
            SemanticResidualOutcome.UNRESOLVED,
            request=request,
            confidence=0.0,
            selected_evidence_ref_ids=(),
            unresolved_reasons=frozenset({LabelUnresolvedReason.MODEL_UNAVAILABLE}),
            model_available=False,
        ),
        audit=_audit(),
    )

    assert result.outcome is SemanticResidualOutcome.UNRESOLVED
    assert result.semantic_evidence == ()
    assert result.unresolved_reasons == frozenset({LabelUnresolvedReason.MODEL_UNAVAILABLE})


def test_fake_runner_fails_closed_for_invalid_fixture_evidence_or_missing_reasons() -> None:
    request = _request()

    with pytest.raises(SemanticResidualPolicyError, match="selected evidence"):
        FakeSemanticResidualRunner().run(
            request,
            fixture=_fixture(
                SemanticResidualOutcome.MATCH,
                request=request,
                selected_evidence_ref_ids=("evidence-ref://semantic/missing",),
            ),
            audit=_audit(),
        )

    with pytest.raises(SemanticResidualPolicyError, match="unresolved reason"):
        FakeSemanticResidualRunner().run(
            request,
            fixture=_fixture(
                SemanticResidualOutcome.ABSTAIN,
                request=request,
                selected_evidence_ref_ids=(),
                unresolved_reasons=frozenset(),
            ),
            audit=_audit(),
        )


def test_missing_safe_evidence_is_abstain_not_positive_label() -> None:
    request = _request()

    result = FakeSemanticResidualRunner().run(
        request,
        fixture=_fixture(
            SemanticResidualOutcome.ABSTAIN,
            request=request,
            confidence=0.0,
            selected_evidence_ref_ids=(),
            unresolved_reasons=frozenset({LabelUnresolvedReason.MISSING_EVIDENCE}),
        ),
        audit=_audit(),
    )

    assert result.outcome is SemanticResidualOutcome.ABSTAIN
    assert result.semantic_evidence == ()
    assert result.unresolved_reasons == frozenset({LabelUnresolvedReason.MISSING_EVIDENCE})


def test_request_and_result_ids_ignore_audit_time_and_hash_seed(tmp_path: Path) -> None:
    request_one = _request(audit=_audit())
    request_two = _request(audit=_audit(datetime(2026, 7, 25, tzinfo=UTC)))
    fixture = _fixture(SemanticResidualOutcome.MATCH, request=request_one)
    result_one = FakeSemanticResidualRunner().run(request_one, fixture=fixture, audit=_audit())
    result_two = FakeSemanticResidualRunner().run(
        request_one,
        fixture=fixture,
        audit=_audit(datetime(2026, 7, 25, tzinfo=UTC)),
    )
    script = tmp_path / "check_semantic_seed.py"
    script.write_text(
        f"""
from datetime import UTC, datetime
from eval_factory.contracts.core import ContractAudit, EvidencePolarity, EvidenceRef, ObjectRef, SourceSpanRef, VersionBinding
from eval_factory.contracts.labeling_v2 import LabelSpecV2, SemanticResidualSpecV2
from eval_factory.contracts.safety import EvidenceBundle
from eval_factory.contracts.trace import CapabilityStatus, TraceCapability
from eval_factory.labeling import FakeSemanticResidualFixture, FakeSemanticResidualRunner, SemanticResidualOutcome, SemanticResidualRequestBuilder, StructuredFactSet, StructuredPredicateCompiler

audit = ContractAudit(created_at=datetime(2026, 7, 24, tzinfo=UTC), created_by='seed-check', governing_versions=(VersionBinding(component='semantic-residual', version='r3-03'),))
trace_ref = ObjectRef(object_type='trace-envelope', object_id='{TRACE_IR_VERSION_ID}', object_version='stored-manifest/v1', object_sha256='{HASH}')
residual = SemanticResidualSpecV2(residual_id='semantic-residual://contextual-recovery/v2', question='Did the later action materially adapt strategy?', evidence_bundle_purpose='label-contextual-recovery', allowed_evidence_types=('user-text',), abstain_conditions=('ambiguous',), model_profile='internal-semantic-labeler-v1', prompt_version='contextual-recovery-label-prompt/v1')
spec = LabelSpecV2(label_spec_id='label-spec://semantic-residual-test/v2', label_version='v2', name='semantic-residual-test', requirement='Semantic residual test label.', prerequisite_predicates=(), positive_predicates=(), negative_predicates=(), semantic_residual=residual, decision_threshold=0.85, review_threshold=0.70, label_plan_ref=None, policy_version='labeling/r3-03-v1', label_spec_sha256='{HASH}', audit=audit)
fact_set = StructuredFactSet(source_trace_id='{SOURCE_TRACE_ID}', trace_ir_version_id='{TRACE_IR_VERSION_ID}', trace_envelope_ref=trace_ref, capabilities=(TraceCapability(capability='tool_events', status=CapabilityStatus.COMPLETE),), audit=audit)
structured = StructuredPredicateCompiler().compile(label_spec=spec, fact_set=fact_set, audit=audit)
subject_ref = ObjectRef(object_type='file-version-projection', object_id='evidence-view://semantic/safe', object_version='v1', object_sha256='{HASH}')
evidence = EvidenceRef(evidence_ref_id='evidence-ref://semantic/safe', subject_ref=subject_ref, source_spans=(SourceSpanRef(span_id='source-span://semantic/safe', source_trace_id='{SOURCE_TRACE_ID}', raw_sha256='{HASH}'),), polarity=EvidencePolarity.POSITIVE, capability='semantic-evidence', capability_complete=True)
bundle = EvidenceBundle(evidence_bundle_id='evidence-bundle://semantic/safe', source_trace_id='{SOURCE_TRACE_ID}', trace_ir_version_id='{TRACE_IR_VERSION_ID}', consumer_stage='semantic-labeler', purpose='label-contextual-recovery', projection_policy_ref=ObjectRef(object_type='projection-policy', object_id='projection-policy://semantic/safe', object_version='v1', object_sha256='{HASH}'), evidence=(evidence,), excluded_subject_refs=(), returned_characters=120, max_characters=1000, tainted_content_included=False, bundle_sha256='{OTHER_HASH}', audit=audit)
request = SemanticResidualRequestBuilder().build(label_spec=spec, structured_result=structured, evidence_bundle=bundle, audit=audit)
fixture = FakeSemanticResidualFixture(fixture_id='semantic-fixture://match', outcome=SemanticResidualOutcome.MATCH, confidence=0.91, selected_evidence_ref_ids=(evidence.evidence_ref_id,), unresolved_reasons=frozenset(), explanation_code='semantic-fixture-result', model_available=True)
result = FakeSemanticResidualRunner().run(request, fixture=fixture, audit=audit)
print(request.semantic_residual_request_id)
print(request.request_sha256)
print(result.semantic_residual_result_id)
print(result.result_sha256)
""",
        encoding="utf-8",
    )
    run_one = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env={**os.environ, "PYTHONHASHSEED": "1"},
        check=True,
        text=True,
        capture_output=True,
    )
    run_two = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env={**os.environ, "PYTHONHASHSEED": "2"},
        check=True,
        text=True,
        capture_output=True,
    )

    assert request_one.semantic_residual_request_id == request_two.semantic_residual_request_id
    assert request_one.request_sha256 == request_two.request_sha256
    assert result_one.semantic_residual_result_id == result_two.semantic_residual_result_id
    assert result_one.result_sha256 == result_two.result_sha256
    assert run_one.stdout == run_two.stdout


def test_semantic_models_reject_raw_or_hidden_payload_fields() -> None:
    request = _request()
    result = FakeSemanticResidualRunner().run(
        request,
        fixture=_fixture(SemanticResidualOutcome.MATCH, request=request),
        audit=_audit(),
    )

    with pytest.raises(ValidationError):
        SemanticResidualRequest.model_validate({**request.model_dump(mode="json"), "raw_trace_text": "deny"})
    with pytest.raises(ValidationError):
        SemanticResidualRequest.model_validate({**request.model_dump(mode="json"), "grader_rules": []})
    with pytest.raises(ValidationError):
        SemanticResidualResult.model_validate({**result.model_dump(mode="json"), "final_output": "deny"})
