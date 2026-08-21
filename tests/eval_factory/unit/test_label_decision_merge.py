from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionValueV2,
    LabelExecutionStatus,
    LabelSpecV2,
    LabelUnresolvedReason,
    PredicateOperatorV2,
    SemanticResidualSpecV2,
    StructuredPredicateV2,
)
from eval_factory.labeling import (
    LabelDecisionMergePolicyError,
    LabelDecisionMerger,
    LabelDecisionMergeRequest,
    LabelDecisionRoute,
    LabelDecisionRoutingPolicy,
    SemanticResidualOutcome,
    SemanticResidualResult,
    StructuredLabelResult,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
ROOT = Path(__file__).resolve().parents[3]


def _audit(created_at: datetime = datetime(2026, 7, 24, tzinfo=UTC)) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="label-decision-merge-test",
        governing_versions=(VersionBinding(component="label-decision-merge", version="r3-04"),),
    )


def _ref(object_type: str, object_id: str | None = None, digest: str = HASH) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=object_id or f"{object_type}://merge-test",
        object_version="v1",
        object_sha256=digest,
    )


def _span() -> SourceSpanRef:
    return SourceSpanRef(
        span_id="source-span://merge-test",
        source_trace_id="source-trace://merge-test",
        raw_sha256=HASH,
    )


def _evidence(
    evidence_id: str,
    *,
    polarity: EvidencePolarity = EvidencePolarity.POSITIVE,
    object_type: str = "trace-event",
    complete: bool = True,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=evidence_id,
        subject_ref=_ref(object_type, f"{object_type}://{evidence_id.rsplit('/', 1)[-1]}"),
        source_spans=(_span(),),
        polarity=polarity,
        capability="merge-evidence",
        capability_complete=complete,
    )


def _predicate(predicate_id: str = "predicate://merge/positive") -> StructuredPredicateV2:
    return StructuredPredicateV2(
        predicate_id=predicate_id,
        fact_type="tool-call-record",
        field_path="tool_family",
        operator=PredicateOperatorV2.EXISTS,
        expected_value=None,
        required_capability="tool_events",
        rule_version="structured-label/r3-04-test",
    )


def _semantic_residual() -> SemanticResidualSpecV2:
    return SemanticResidualSpecV2(
        residual_id="semantic-residual://merge-test",
        question="Does the bounded evidence satisfy the residual label?",
        evidence_bundle_purpose="label-merge",
        allowed_evidence_types=("trace-event",),
        abstain_conditions=("ambiguous",),
        model_profile="internal-semantic-labeler-v1",
        prompt_version="semantic-merge-prompt/v1",
    )


def _label_spec(
    *,
    semantic: bool,
    decision_threshold: float = 0.8,
    review_threshold: float = 0.6,
) -> LabelSpecV2:
    return LabelSpecV2(
        label_spec_id="label-spec://merge-test/v2",
        label_version="v2",
        name="merge-test",
        requirement="Merge test label.",
        prerequisite_predicates=(),
        positive_predicates=() if semantic else (_predicate(),),
        negative_predicates=(),
        semantic_residual=_semantic_residual() if semantic else None,
        decision_threshold=decision_threshold,
        review_threshold=review_threshold,
        label_plan_ref=None,
        policy_version="labeling/r3-04-test",
        label_spec_sha256=HASH,
        audit=_audit(),
    )


def _label_spec_ref(spec: LabelSpecV2) -> ObjectRef:
    return ObjectRef(
        object_type="label-spec",
        object_id=spec.label_spec_id,
        object_version=spec.label_version,
        object_sha256=spec.label_spec_sha256,
    )


def _trace_ref() -> ObjectRef:
    return _ref("trace-envelope", "trace-envelope://merge-test", OTHER_HASH)


def _structured_result(
    spec: LabelSpecV2,
    *,
    positive: tuple[EvidenceRef, ...] = (),
    negative: tuple[EvidenceRef, ...] = (),
    complete: bool = True,
    semantic_required: bool = False,
    unresolved: frozenset[LabelUnresolvedReason] = frozenset(),
) -> StructuredLabelResult:
    return StructuredLabelResult(
        structured_label_result_id="structured-label-result://merge-test",
        label_spec_ref=_label_spec_ref(spec),
        trace_envelope_ref=_trace_ref(),
        prerequisite_results=(),
        positive_results=(),
        negative_results=(),
        prerequisites_satisfied=True,
        positive_evidence=positive,
        negative_evidence=negative,
        structured_capability_complete=complete,
        semantic_evaluation_required=semantic_required,
        unresolved_reasons=unresolved,
        uncertainties=(),
        result_sha256=HASH,
        audit=_audit(),
    )


def _semantic_result(
    spec: LabelSpecV2,
    *,
    outcome: SemanticResidualOutcome,
    confidence: float,
    evidence: tuple[EvidenceRef, ...] = (),
    unresolved: frozenset[LabelUnresolvedReason] = frozenset(),
) -> SemanticResidualResult:
    return SemanticResidualResult(
        semantic_residual_result_id=f"semantic-residual-result://{outcome.value.lower()}",
        semantic_residual_request_ref=_ref(
            "semantic-residual-request",
            "semantic-residual-request://merge-test",
            OTHER_HASH,
        ),
        label_spec_ref=_label_spec_ref(spec),
        trace_envelope_ref=_trace_ref(),
        outcome=outcome,
        confidence=confidence,
        semantic_evidence=evidence,
        model_profile="internal-semantic-labeler-v1",
        prompt_version="semantic-merge-prompt/v1",
        unresolved_reasons=unresolved,
        explanation_code="merge-test",
        result_sha256=OTHER_HASH,
        audit=_audit(),
    )


def _merge(
    spec: LabelSpecV2,
    structured: StructuredLabelResult,
    *,
    semantic: SemanticResidualResult | None = None,
    routing_policy: LabelDecisionRoutingPolicy | None = None,
    audit: ContractAudit | None = None,
):
    return LabelDecisionMerger().merge(
        LabelDecisionMergeRequest(
            label_spec=spec,
            structured_result=structured,
            semantic_result=semantic,
            routing_policy=routing_policy or LabelDecisionRoutingPolicy(),
            audit=audit or _audit(),
        )
    )


def test_structured_positive_evidence_emits_final_match() -> None:
    spec = _label_spec(semantic=False)
    structured = _structured_result(
        spec,
        positive=(_evidence("evidence-ref://merge/positive"),),
    )

    result = _merge(spec, structured)

    assert result.route is LabelDecisionRoute.FINAL
    assert result.label_decision.decision is LabelDecisionValueV2.MATCH
    assert result.label_decision.execution_status is LabelExecutionStatus.FINAL
    assert result.label_decision.positive_evidence == structured.positive_evidence
    assert result.label_decision.confidence == 1.0


def test_structured_negative_evidence_emits_final_no_match() -> None:
    spec = _label_spec(semantic=False)
    structured = _structured_result(
        spec,
        negative=(
            _evidence(
                "evidence-ref://merge/negative",
                polarity=EvidencePolarity.NEGATIVE,
            ),
        ),
    )

    result = _merge(spec, structured)

    assert result.route is LabelDecisionRoute.FINAL
    assert result.label_decision.decision is LabelDecisionValueV2.NO_MATCH
    assert result.label_decision.execution_status is LabelExecutionStatus.FINAL
    assert result.label_decision.negative_evidence == structured.negative_evidence
    assert result.label_decision.structured_capability_complete is True


def test_incomplete_structured_capability_abstains_unresolved() -> None:
    spec = _label_spec(semantic=False)
    structured = _structured_result(
        spec,
        complete=False,
        unresolved=frozenset({LabelUnresolvedReason.INCOMPLETE_STRUCTURED_CAPABILITY}),
    )

    result = _merge(spec, structured)

    assert result.route is LabelDecisionRoute.TYPED_UNRESOLVED_QUEUE
    assert result.label_decision.decision is LabelDecisionValueV2.ABSTAIN
    assert result.label_decision.execution_status is LabelExecutionStatus.UNRESOLVED
    assert result.label_decision.unresolved_reasons == frozenset(
        {LabelUnresolvedReason.INCOMPLETE_STRUCTURED_CAPABILITY}
    )


def test_missing_structured_evidence_abstains_with_missing_evidence() -> None:
    spec = _label_spec(semantic=False)
    structured = _structured_result(spec)

    result = _merge(spec, structured)

    assert result.label_decision.decision is LabelDecisionValueV2.ABSTAIN
    assert LabelUnresolvedReason.MISSING_EVIDENCE in result.label_decision.unresolved_reasons


def test_semantic_high_confidence_match_emits_final_match_with_model_metadata() -> None:
    spec = _label_spec(semantic=True)
    structured = _structured_result(spec, semantic_required=True)
    semantic_evidence = (_evidence("evidence-ref://merge/semantic"),)
    semantic = _semantic_result(
        spec,
        outcome=SemanticResidualOutcome.MATCH,
        confidence=0.91,
        evidence=semantic_evidence,
    )

    result = _merge(spec, structured, semantic=semantic)

    assert result.route is LabelDecisionRoute.FINAL
    assert result.label_decision.decision is LabelDecisionValueV2.MATCH
    assert result.label_decision.semantic_evidence == semantic_evidence
    assert result.label_decision.model_profile == "internal-semantic-labeler-v1"
    assert result.label_decision.prompt_version == "semantic-merge-prompt/v1"
    assert result.label_decision.confidence == 0.91


def test_semantic_no_match_without_structured_negative_evidence_abstains() -> None:
    spec = _label_spec(semantic=True)
    structured = _structured_result(spec, semantic_required=True)
    semantic = _semantic_result(
        spec,
        outcome=SemanticResidualOutcome.NO_MATCH,
        confidence=0.93,
        evidence=(_evidence("evidence-ref://merge/semantic-no-match"),),
    )

    result = _merge(spec, structured, semantic=semantic)

    assert result.label_decision.decision is LabelDecisionValueV2.ABSTAIN
    assert LabelUnresolvedReason.MISSING_EVIDENCE in result.label_decision.unresolved_reasons


def test_semantic_low_confidence_can_route_to_user_inspection() -> None:
    spec = _label_spec(semantic=True, decision_threshold=0.8, review_threshold=0.6)
    structured = _structured_result(spec, semantic_required=True)
    semantic = _semantic_result(
        spec,
        outcome=SemanticResidualOutcome.MATCH,
        confidence=0.7,
        evidence=(_evidence("evidence-ref://merge/low-confidence"),),
    )

    result = _merge(
        spec,
        structured,
        semantic=semantic,
        routing_policy=LabelDecisionRoutingPolicy(route_low_confidence_to_user_inspection=True),
    )

    assert result.route is LabelDecisionRoute.USER_INSPECTION_QUEUE
    assert result.label_decision.execution_status is LabelExecutionStatus.REVIEW_REQUIRED
    assert LabelUnresolvedReason.LOW_CONFIDENCE in result.label_decision.unresolved_reasons
    assert LabelUnresolvedReason.USER_INSPECTION_REQUIRED in result.label_decision.unresolved_reasons


def test_semantic_below_review_threshold_routes_to_typed_unresolved() -> None:
    spec = _label_spec(semantic=True, decision_threshold=0.8, review_threshold=0.6)
    structured = _structured_result(spec, semantic_required=True)
    semantic = _semantic_result(
        spec,
        outcome=SemanticResidualOutcome.MATCH,
        confidence=0.5,
        evidence=(_evidence("evidence-ref://merge/below-review"),),
    )

    result = _merge(spec, structured, semantic=semantic)

    assert result.route is LabelDecisionRoute.TYPED_UNRESOLVED_QUEUE
    assert result.label_decision.execution_status is LabelExecutionStatus.UNRESOLVED
    assert result.label_decision.unresolved_reasons == frozenset({LabelUnresolvedReason.LOW_CONFIDENCE})


def test_semantic_unresolved_model_unavailable_propagates_reason_without_fabrication() -> None:
    spec = _label_spec(semantic=True)
    structured = _structured_result(spec, semantic_required=True)
    semantic = _semantic_result(
        spec,
        outcome=SemanticResidualOutcome.UNRESOLVED,
        confidence=0.0,
        unresolved=frozenset({LabelUnresolvedReason.MODEL_UNAVAILABLE}),
    )

    result = _merge(spec, structured, semantic=semantic)

    assert result.label_decision.decision is LabelDecisionValueV2.ABSTAIN
    assert result.label_decision.semantic_evidence == ()
    assert result.label_decision.unresolved_reasons == frozenset({LabelUnresolvedReason.MODEL_UNAVAILABLE})


def test_structured_semantic_disagreement_abstains_with_conflict_reason() -> None:
    spec = _label_spec(semantic=True)
    structured = _structured_result(
        spec,
        positive=(_evidence("evidence-ref://merge/structured-positive"),),
        semantic_required=True,
    )
    semantic = _semantic_result(
        spec,
        outcome=SemanticResidualOutcome.NO_MATCH,
        confidence=0.95,
        evidence=(_evidence("evidence-ref://merge/semantic-disagrees"),),
    )

    result = _merge(spec, structured, semantic=semantic)

    assert result.label_decision.decision is LabelDecisionValueV2.ABSTAIN
    assert LabelUnresolvedReason.CONFLICTING_EVIDENCE in result.label_decision.unresolved_reasons


def test_rejects_semantic_result_when_semantic_evaluation_not_required() -> None:
    spec = _label_spec(semantic=False)
    structured = _structured_result(spec, positive=(_evidence("evidence-ref://merge/positive"),))
    semantic = _semantic_result(
        _label_spec(semantic=True),
        outcome=SemanticResidualOutcome.MATCH,
        confidence=0.9,
        evidence=(_evidence("evidence-ref://merge/semantic"),),
    )

    with pytest.raises(LabelDecisionMergePolicyError, match="not required"):
        _merge(spec, structured, semantic=semantic)


def test_unsafe_evidence_subject_refs_fail_closed() -> None:
    spec = _label_spec(semantic=False)
    structured = _structured_result(
        spec,
        positive=(
            _evidence(
                "evidence-ref://merge/unsafe",
                object_type="final-output",
            ),
        ),
    )

    with pytest.raises(LabelDecisionMergePolicyError, match="unsafe"):
        _merge(spec, structured)


def test_decision_ids_ignore_audit_time_and_hash_seed(tmp_path: Path) -> None:
    spec = _label_spec(semantic=False)
    structured = _structured_result(spec, positive=(_evidence("evidence-ref://merge/positive"),))
    first = _merge(spec, structured, audit=_audit())
    second = _merge(spec, structured, audit=_audit(datetime(2026, 7, 25, tzinfo=UTC)))
    script = tmp_path / "check_decision_merge_seed.py"
    script.write_text(
        f"""
from datetime import UTC, datetime
from eval_factory.contracts.core import ContractAudit, EvidencePolarity, EvidenceRef, ObjectRef, SourceSpanRef, VersionBinding
from eval_factory.contracts.labeling_v2 import LabelSpecV2, PredicateOperatorV2, StructuredPredicateV2
from eval_factory.labeling import LabelDecisionMergeRequest, LabelDecisionMerger, LabelDecisionRoutingPolicy, StructuredLabelResult

HASH = '{HASH}'
OTHER_HASH = '{OTHER_HASH}'
audit = ContractAudit(created_at=datetime(2026, 7, 24, tzinfo=UTC), created_by='seed-check', governing_versions=(VersionBinding(component='label-decision-merge', version='r3-04'),))
predicate = StructuredPredicateV2(predicate_id='predicate://merge/positive', fact_type='tool-call-record', field_path='tool_family', operator=PredicateOperatorV2.EXISTS, expected_value=None, required_capability='tool_events', rule_version='structured-label/r3-04-test')
spec = LabelSpecV2(label_spec_id='label-spec://merge-test/v2', label_version='v2', name='merge-test', requirement='Merge test label.', prerequisite_predicates=(), positive_predicates=(predicate,), negative_predicates=(), semantic_residual=None, decision_threshold=0.8, review_threshold=0.6, label_plan_ref=None, policy_version='labeling/r3-04-test', label_spec_sha256=HASH, audit=audit)
span = SourceSpanRef(span_id='source-span://merge-test', source_trace_id='source-trace://merge-test', raw_sha256=HASH)
evidence = EvidenceRef(evidence_ref_id='evidence-ref://merge/positive', subject_ref=ObjectRef(object_type='trace-event', object_id='trace-event://positive', object_version='v1', object_sha256=HASH), source_spans=(span,), polarity=EvidencePolarity.POSITIVE, capability='merge-evidence', capability_complete=True)
structured = StructuredLabelResult(structured_label_result_id='structured-label-result://merge-test', label_spec_ref=ObjectRef(object_type='label-spec', object_id=spec.label_spec_id, object_version=spec.label_version, object_sha256=HASH), trace_envelope_ref=ObjectRef(object_type='trace-envelope', object_id='trace-envelope://merge-test', object_version='v1', object_sha256=OTHER_HASH), prerequisite_results=(), positive_results=(), negative_results=(), prerequisites_satisfied=True, positive_evidence=(evidence,), negative_evidence=(), structured_capability_complete=True, semantic_evaluation_required=False, unresolved_reasons=frozenset(), uncertainties=(), result_sha256=HASH, audit=audit)
result = LabelDecisionMerger().merge(LabelDecisionMergeRequest(label_spec=spec, structured_result=structured, semantic_result=None, routing_policy=LabelDecisionRoutingPolicy(), audit=audit))
print(result.label_decision.label_decision_id)
print(result.label_decision.decision_sha256)
print(result.label_decision_merge_result_id)
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

    assert first.label_decision.label_decision_id == second.label_decision.label_decision_id
    assert first.label_decision.decision_sha256 == second.label_decision.decision_sha256
    assert first.label_decision_merge_result_id == second.label_decision_merge_result_id
    assert first.result_sha256 == second.result_sha256
    assert run_one.stdout == run_two.stdout
