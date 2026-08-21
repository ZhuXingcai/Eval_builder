from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts import (
    LabelDecisionV2,
    LabelDecisionValueV2,
    LabelExecutionStatus,
    LabelSpecV2,
    LabelUnresolvedReason,
    SelectionContextV2,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.core_v2 import require_matching_hash_v2
from eval_factory.contracts.labeling import (
    LabelDecision,
    LabelDecisionValue,
    PredicateOperator,
    ReviewStatus,
)
from eval_factory.contracts.labeling_v2 import (
    PredicateOperatorV2,
    SemanticResidualSpecV2,
    StructuredPredicateV2,
)

ROOT = Path(__file__).resolve().parents[3]
HASH = "a" * 64
OTHER_HASH = "b" * 64


def _audit(created_at: datetime = datetime(2026, 7, 24, tzinfo=UTC)) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="v2-labeling-test",
        governing_versions=(VersionBinding(component="labeling-v2", version="r3-01", sha256=HASH),),
    )


def _ref(name: str, *, digest: str = HASH, version: str = "v2") -> ObjectRef:
    return ObjectRef(
        object_type=name,
        object_id=f"{name}://example/{version}",
        object_version=version,
        object_sha256=digest,
    )


def _span() -> SourceSpanRef:
    return SourceSpanRef(
        span_id="source-span://labeling/v2",
        source_trace_id="source-trace://labeling/v2",
        raw_sha256=HASH,
    )


def _evidence(
    evidence_id: str = "evidence-ref://labeling/positive",
    *,
    complete: bool = True,
) -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=evidence_id,
        subject_ref=_ref("trace-event"),
        source_spans=(_span(),),
        polarity=EvidencePolarity.POSITIVE,
        capability="trace-query",
        capability_complete=complete,
    )


def _predicate(predicate_id: str = "predicate://powershell-error") -> StructuredPredicateV2:
    return StructuredPredicateV2(
        predicate_id=predicate_id,
        fact_type="tool-call-record",
        field_path="tool_family",
        operator=PredicateOperatorV2.EQUALS,
        expected_value="powershell",
        required_capability="tool-events",
        rule_version="structured-label/r3-01",
    )


def _semantic() -> SemanticResidualSpecV2:
    return SemanticResidualSpecV2(
        residual_id="semantic-residual://contextual-behavior",
        question="Does the trace show contextual troubleshooting behavior?",
        evidence_bundle_purpose="labeling",
        allowed_evidence_types=("interaction-segment", "tool-call-record"),
        abstain_conditions=("Evidence is ambiguous",),
        model_profile="internal-semantic-fake",
        prompt_version="semantic-label/r3-01",
    )


def _label_spec(**updates: object) -> LabelSpecV2:
    base = {
        "label_spec_id": "label-spec://powershell-error/v2",
        "label_version": "v2",
        "name": "PowerShell error signature",
        "requirement": "Trace contains an executed PowerShell command that failed.",
        "prerequisite_predicates": (_predicate("predicate://tool-exists"),),
        "positive_predicates": (_predicate("predicate://powershell-error"),),
        "negative_predicates": (),
        "semantic_residual": _semantic(),
        "decision_threshold": 0.8,
        "review_threshold": 0.6,
        "label_plan_ref": _ref("label-plan"),
        "policy_version": "labeling/r3-01-v1",
        "label_spec_sha256": HASH,
        "audit": _audit(),
    }
    base.update(updates)
    return LabelSpecV2(**base)


def _decision(**updates: object) -> LabelDecisionV2:
    base = {
        "label_decision_id": "label-decision://trace-label/v2",
        "label_spec_ref": _ref("label-spec"),
        "trace_envelope_ref": _ref("trace-envelope"),
        "decision": LabelDecisionValueV2.MATCH,
        "execution_status": LabelExecutionStatus.FINAL,
        "positive_evidence": (_evidence(),),
        "negative_evidence": (),
        "semantic_evidence": (),
        "structured_capability_complete": True,
        "confidence": 0.91,
        "rule_version": "structured-label/r3-01",
        "model_profile": None,
        "prompt_version": None,
        "unresolved_reasons": frozenset(),
        "policy_version": "labeling/r3-01-v1",
        "decision_sha256": HASH,
        "audit": _audit(),
    }
    base.update(updates)
    return LabelDecisionV2(**base)


def test_label_spec_v2_accepts_mixed_structured_and_semantic_plan() -> None:
    spec = _label_spec()

    assert spec.schema_version == "eval-factory/label-spec/v2"
    assert spec.label_plan_ref is not None
    assert spec.semantic_residual is not None
    assert spec.positive_predicates[0].required_capability == "tool-events"


def test_not_error_signature_is_additive_to_v2_only() -> None:
    predicate = StructuredPredicateV2(
        predicate_id="predicate://powershell-error/negative/r8-10-v2",
        fact_type="tool-call-record",
        field_path="normalized_error_signature",
        operator=PredicateOperatorV2.NOT_ERROR_SIGNATURE,
        expected_value="powershell:*:paired_error",
        required_capability="call_result_pairing",
        rule_version="powershell-error-signature/r8-10-v2",
    )

    assert predicate.operator is PredicateOperatorV2.NOT_ERROR_SIGNATURE
    assert "NOT_ERROR_SIGNATURE" not in {operator.value for operator in PredicateOperator}


def test_label_spec_v2_rejects_empty_duplicate_and_invalid_thresholds() -> None:
    with pytest.raises(ValidationError, match="at least one structured predicate or semantic residual"):
        _label_spec(
            prerequisite_predicates=(),
            positive_predicates=(),
            negative_predicates=(),
            semantic_residual=None,
        )

    duplicate = _predicate("predicate://same")
    with pytest.raises(ValidationError, match="predicate IDs must be unique"):
        _label_spec(prerequisite_predicates=(duplicate,), positive_predicates=(duplicate,))

    with pytest.raises(ValidationError, match="review threshold cannot exceed decision threshold"):
        _label_spec(review_threshold=0.9, decision_threshold=0.7)


def test_label_spec_v2_requires_label_plan_ref_to_reference_label_plan() -> None:
    with pytest.raises(ValidationError, match="label_plan_ref must reference label-plan"):
        _label_spec(label_plan_ref=_ref("task-rewrite-plan"))


def test_label_decision_v2_match_requires_positive_or_semantic_evidence() -> None:
    with pytest.raises(ValidationError, match="MATCH requires positive or semantic evidence"):
        _decision(positive_evidence=(), semantic_evidence=())


def test_label_decision_v2_no_match_requires_complete_negative_evidence() -> None:
    with pytest.raises(ValidationError, match="NO_MATCH requires complete structured capability"):
        _decision(
            decision=LabelDecisionValueV2.NO_MATCH,
            positive_evidence=(),
            negative_evidence=(_evidence("evidence-ref://negative"),),
            structured_capability_complete=False,
        )

    with pytest.raises(ValidationError, match="NO_MATCH evidence must be capability-complete"):
        _decision(
            decision=LabelDecisionValueV2.NO_MATCH,
            positive_evidence=(),
            negative_evidence=(_evidence("evidence-ref://negative", complete=False),),
        )


def test_label_decision_v2_semantic_evidence_requires_model_and_prompt_versions() -> None:
    with pytest.raises(ValidationError, match="semantic evidence requires model_profile and prompt_version"):
        _decision(
            positive_evidence=(),
            semantic_evidence=(_evidence("evidence-ref://semantic"),),
            model_profile=None,
            prompt_version=None,
        )

    accepted = _decision(
        positive_evidence=(),
        semantic_evidence=(_evidence("evidence-ref://semantic"),),
        model_profile="internal-semantic-fake",
        prompt_version="semantic-label/r3-01",
    )
    assert accepted.decision is LabelDecisionValueV2.MATCH


def test_label_decision_v2_abstain_and_unresolved_require_reason_codes() -> None:
    with pytest.raises(ValidationError, match="ABSTAIN requires unresolved reason"):
        _decision(
            decision=LabelDecisionValueV2.ABSTAIN,
            positive_evidence=(),
            execution_status=LabelExecutionStatus.REVIEW_REQUIRED,
        )

    with pytest.raises(ValidationError, match="UNRESOLVED status requires unresolved reason"):
        _decision(
            execution_status=LabelExecutionStatus.UNRESOLVED,
            unresolved_reasons=frozenset(),
        )

    with pytest.raises(ValidationError, match="FINAL decisions cannot carry unresolved reasons"):
        _decision(unresolved_reasons=frozenset({LabelUnresolvedReason.CONFLICTING_EVIDENCE}))

    accepted = _decision(
        decision=LabelDecisionValueV2.ABSTAIN,
        execution_status=LabelExecutionStatus.UNRESOLVED,
        positive_evidence=(),
        unresolved_reasons=frozenset({LabelUnresolvedReason.AMBIGUOUS_EVIDENCE}),
    )
    assert accepted.decision is LabelDecisionValueV2.ABSTAIN


def test_selection_context_v2_rejects_unsafe_selection_material() -> None:
    base = {
        "selection_context_id": "selection-context://safe/v2",
        "candidate_id": "candidate://safe/v2",
        "approved_label_decision_refs": (_ref("label-decision"),),
        "safe_evidence_bundle_ref": _ref("evidence-bundle"),
        "task_authoring_note_refs": (),
        "excluded_signal_hashes": (HASH,),
        "projection_policy_ref": _ref("projection-policy"),
        "policy_version": "selection-context/r3-01-v1",
        "selection_context_sha256": HASH,
        "audit": _audit(),
    }
    safe = SelectionContextV2(**base)
    assert safe.approved_label_decision_refs[0].object_type == "label-decision"

    for unsafe_ref in (
        _ref("raw-trace"),
        _ref("private-reference"),
        _ref("grader-rule"),
        _ref("hidden-selection-signal"),
        _ref("final-output"),
    ):
        with pytest.raises(ValidationError, match="unsafe selection context reference"):
            SelectionContextV2(**{**base, "task_authoring_note_refs": (unsafe_ref,)})

    with pytest.raises(ValidationError, match="approved_label_decision_refs must reference label-decision"):
        SelectionContextV2(**{**base, "approved_label_decision_refs": (_ref("trace-event"),)})


def test_labeling_v2_ids_and_carried_hashes_ignore_audit_timestamp() -> None:
    first = _label_spec(audit=_audit(datetime(2026, 7, 24, tzinfo=UTC)))
    second = _label_spec(audit=_audit(datetime(2026, 7, 25, tzinfo=UTC)))

    assert first.label_spec_id == second.label_spec_id
    assert first.label_spec_sha256 == second.label_spec_sha256
    assert first.canonical_sha256() != second.canonical_sha256()


def test_labeling_v2_hashing_is_stable_across_python_hash_seed(tmp_path: Path) -> None:
    script = tmp_path / "check_labeling_seed.py"
    script.write_text(
        """
from datetime import UTC, datetime
from eval_factory.contracts import LabelDecisionV2, LabelDecisionValueV2, LabelExecutionStatus, LabelUnresolvedReason
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding

def ref(name):
    return ObjectRef(object_type=name, object_id=f'{name}://example/v2', object_version='v2', object_sha256='a' * 64)

audit = ContractAudit(
    created_at=datetime(2026, 7, 24, tzinfo=UTC),
    created_by='seed-test',
    governing_versions=(VersionBinding(component='labeling-v2', version='r3-01', sha256='a' * 64),),
)
decision = LabelDecisionV2(
    label_decision_id='label-decision://trace-label/v2',
    label_spec_ref=ref('label-spec'),
    trace_envelope_ref=ref('trace-envelope'),
    decision=LabelDecisionValueV2.ABSTAIN,
    execution_status=LabelExecutionStatus.UNRESOLVED,
    positive_evidence=(),
    negative_evidence=(),
    semantic_evidence=(),
    structured_capability_complete=False,
    confidence=0.4,
    unresolved_reasons=frozenset({LabelUnresolvedReason.AMBIGUOUS_EVIDENCE, LabelUnresolvedReason.LOW_CONFIDENCE}),
    policy_version='labeling/r3-01-v1',
    decision_sha256='a' * 64,
    audit=audit,
)
print(decision.canonical_sha256())
""",
        encoding="utf-8",
    )
    outputs = []
    for seed in ("1", "99"):
        result = subprocess.run(
            [sys.executable, str(script)],
            check=False,
            capture_output=True,
            text=True,
            env={**dict(PYTHONHASHSEED=seed), "PYTHONPATH": "src"},
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.strip())

    assert len(set(outputs)) == 1


def test_labeling_v2_models_reject_unknown_raw_trace_fields_and_stale_refs() -> None:
    spec = _label_spec()
    with pytest.raises(ValidationError):
        LabelSpecV2.model_validate({**spec.model_dump(mode="json"), "raw_trace_text": "forbidden"})

    stale = ObjectRef(
        object_type="label-spec",
        object_id="label-spec://stale/v2",
        object_version="v2",
        object_sha256=OTHER_HASH,
    )
    with pytest.raises(ValueError, match="stale or mismatched v2 object reference"):
        require_matching_hash_v2(stale, spec)


def test_v1_label_decision_behavior_remains_compatible() -> None:
    with pytest.raises(ValidationError, match="NO_MATCH requires complete structured capability"):
        LabelDecision(
            label_decision_id="label-decision://example/v1",
            label_spec_ref=_ref("label-spec", version="v1"),
            trace_envelope_ref=_ref("trace-envelope", version="v1"),
            decision=LabelDecisionValue.NO_MATCH,
            structured_capability_complete=False,
            confidence=0.9,
            review_status=ReviewStatus.REVIEW_REQUIRED,
            audit=_audit(),
        )
