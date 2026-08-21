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
    LabelDecisionV2,
    LabelDecisionValueV2,
    LabelExecutionStatus,
    LabelUnresolvedReason,
)
from eval_factory.labeling import (
    LABEL_CALIBRATION_POLICY_VERSION,
    LabelBatchExporter,
    LabelBatchPlan,
    LabelBatchProgressTracker,
    LabelCalibrationClaimScope,
    LabelCalibrationFindingCode,
    LabelCalibrationInput,
    LabelCalibrationOutcome,
    LabelCalibrationPolicyError,
    LabelCalibrationReference,
    LabelCalibrationReviewStatus,
    LabelCalibrationRunner,
    LabelCalibrationSemanticResidualStatus,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
THIRD_HASH = "c" * 64
FOURTH_HASH = "d" * 64
ROOT = Path(__file__).resolve().parents[3]


def _audit(created_at: datetime = datetime(2026, 7, 25, tzinfo=UTC)) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="label-calibration-test",
        governing_versions=(VersionBinding(component="label-calibration", version="r3-06"),),
    )


def _ref(object_type: str, object_id: str, digest: str = HASH) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version="v1",
        object_sha256=digest,
    )


def _trace_ref(index: int = 1) -> ObjectRef:
    return _ref("trace-envelope", f"trace-envelope://calibration/{index}", [HASH, OTHER_HASH][index - 1])


def _label_ref(index: int = 1) -> ObjectRef:
    return _ref("label-spec", f"label-spec://calibration/{index}", [THIRD_HASH, FOURTH_HASH][index - 1])


def _span() -> SourceSpanRef:
    return SourceSpanRef(
        span_id="source-span://calibration/1",
        source_trace_id="source-trace://calibration/1",
        raw_sha256=HASH,
    )


def _evidence(
    evidence_id: str,
    *,
    polarity: EvidencePolarity = EvidencePolarity.POSITIVE,
    object_type: str = "trace-event",
) -> EvidenceRef:
    return EvidenceRef(
        evidence_ref_id=evidence_id,
        subject_ref=_ref(object_type, f"{object_type}://calibration/{evidence_id.rsplit('/', 1)[-1]}"),
        source_spans=(_span(),),
        polarity=polarity,
        capability="calibration-evidence",
        capability_complete=True,
    )


def _decision(
    *,
    trace_ref: ObjectRef | None = None,
    label_ref: ObjectRef | None = None,
    decision_id: str = "label-decision://calibration/1",
    decision: LabelDecisionValueV2 = LabelDecisionValueV2.MATCH,
    positive: tuple[EvidenceRef, ...] = (_evidence("evidence-ref://calibration/positive"),),
    negative: tuple[EvidenceRef, ...] = (),
    semantic: tuple[EvidenceRef, ...] = (),
    complete: bool = True,
    confidence: float = 1.0,
    rule_version: str | None = "structured-labeling/r3-02-v1",
    model_profile: str | None = None,
    prompt_version: str | None = None,
    unresolved: frozenset[LabelUnresolvedReason] = frozenset(),
    decision_sha256: str = OTHER_HASH,
) -> LabelDecisionV2:
    return LabelDecisionV2(
        label_decision_id=decision_id,
        label_spec_ref=label_ref or _label_ref(),
        trace_envelope_ref=trace_ref or _trace_ref(),
        decision=decision,
        execution_status=(
            LabelExecutionStatus.UNRESOLVED
            if decision is LabelDecisionValueV2.ABSTAIN
            else LabelExecutionStatus.FINAL
        ),
        positive_evidence=positive if decision is LabelDecisionValueV2.MATCH else (),
        negative_evidence=negative if decision is LabelDecisionValueV2.NO_MATCH else (),
        semantic_evidence=semantic,
        structured_capability_complete=complete,
        confidence=confidence,
        rule_version=rule_version,
        model_profile=model_profile,
        prompt_version=prompt_version,
        unresolved_reasons=unresolved,
        policy_version="label-decision-merge/r3-04-v1",
        decision_sha256=decision_sha256,
        audit=_audit(),
    )


def _reference(
    *,
    trace_ref: ObjectRef | None = None,
    label_ref: ObjectRef | None = None,
    annotation_id: str = "label-annotation://calibration/1",
    expected: LabelDecisionValueV2 = LabelDecisionValueV2.MATCH,
    positive_ids: tuple[str, ...] = ("evidence-ref://calibration/positive",),
    negative_ids: tuple[str, ...] = (),
    semantic_ids: tuple[str, ...] = (),
    complete: bool = True,
    semantic_status: LabelCalibrationSemanticResidualStatus = (
        LabelCalibrationSemanticResidualStatus.NOT_REQUIRED
    ),
    confidence: float = 1.0,
    rule_version: str | None = "structured-labeling/r3-02-v1",
    model_profile: str | None = None,
    prompt_version: str | None = None,
    review_status: LabelCalibrationReviewStatus = LabelCalibrationReviewStatus.APPROVED,
) -> LabelCalibrationReference:
    return LabelCalibrationReference(
        annotation_ref=_ref("label-annotation", annotation_id, FOURTH_HASH),
        canary_instance_id="LH_005",
        trace_envelope_ref=trace_ref or _trace_ref(),
        label_spec_ref=label_ref or _label_ref(),
        expected_decision=expected,
        positive_evidence_ref_ids=positive_ids,
        negative_evidence_ref_ids=negative_ids,
        semantic_evidence_ref_ids=semantic_ids,
        structured_capability_complete=complete,
        semantic_residual_status=semantic_status,
        confidence=confidence,
        rule_version=rule_version,
        model_profile=model_profile,
        prompt_version=prompt_version,
        review_status=review_status,
        policy_version="eval-factory-annotation-guide/v2",
    )


def _input(
    *,
    references: tuple[LabelCalibrationReference, ...],
    decisions: tuple[LabelDecisionV2, ...] = (),
    export_rows: tuple[dict[str, object], ...] = (),
    batch_export_ref: ObjectRef | None = None,
    audit: ContractAudit | None = None,
) -> LabelCalibrationInput:
    return LabelCalibrationInput.build(
        canary_manifest_ref=_ref(
            "canary-manifest",
            "canary-manifest://eval-factory/v4",
            HASH,
        ),
        annotation_contract_ref=_ref(
            "annotation-contract-manifest",
            "annotation-contract-manifest://eval-factory/v2",
            OTHER_HASH,
        ),
        annotation_guide_ref=_ref(
            "annotation-guide",
            "annotation-guide://eval-factory/v2",
            THIRD_HASH,
        ),
        references=references,
        decisions=decisions,
        export_rows=export_rows,
        batch_export_ref=batch_export_ref,
        audit=audit or _audit(),
    )


def _compare(
    *,
    references: tuple[LabelCalibrationReference, ...],
    decisions: tuple[LabelDecisionV2, ...] = (),
    export_rows: tuple[dict[str, object], ...] = (),
    batch_export_ref: ObjectRef | None = None,
    audit: ContractAudit | None = None,
):
    return LabelCalibrationRunner().compare(
        _input(
            references=references,
            decisions=decisions,
            export_rows=export_rows,
            batch_export_ref=batch_export_ref,
            audit=audit,
        )
    )


def test_structured_decision_exactly_matches_reference() -> None:
    report = _compare(references=(_reference(),), decisions=(_decision(),))

    comparison = report.comparisons[0]
    assert comparison.outcome is LabelCalibrationOutcome.EXACT_MATCH
    assert comparison.finding_codes == frozenset()
    assert report.summary.exact_matches == 1
    assert report.summary.blocking_findings == 0


def test_semantic_decision_matches_evidence_and_model_metadata() -> None:
    semantic = (_evidence("evidence-ref://calibration/semantic"),)
    decision = _decision(
        positive=(),
        semantic=semantic,
        confidence=0.91,
        rule_version="structured-labeling/r3-02-v1",
        model_profile="internal-semantic-labeler-v1",
        prompt_version="contextual-recovery-label-prompt/v1",
    )
    reference = _reference(
        positive_ids=(),
        semantic_ids=("evidence-ref://calibration/semantic",),
        semantic_status=LabelCalibrationSemanticResidualStatus.REQUIRED,
        confidence=0.91,
        model_profile="internal-semantic-labeler-v1",
        prompt_version="contextual-recovery-label-prompt/v1",
    )

    report = _compare(references=(reference,), decisions=(decision,))

    assert report.comparisons[0].outcome is LabelCalibrationOutcome.EXACT_MATCH
    assert report.comparisons[0].observed_model_profile == "internal-semantic-labeler-v1"
    assert report.comparisons[0].observed_prompt_version == "contextual-recovery-label-prompt/v1"


def test_approved_ambiguity_abstention_is_non_blocking() -> None:
    decision = _decision(
        decision=LabelDecisionValueV2.ABSTAIN,
        positive=(),
        complete=False,
        confidence=0.0,
        rule_version=None,
        unresolved=frozenset({LabelUnresolvedReason.AMBIGUOUS_EVIDENCE}),
    )
    reference = _reference(
        expected=LabelDecisionValueV2.ABSTAIN,
        positive_ids=(),
        complete=False,
        confidence=0.0,
        rule_version=None,
        review_status=LabelCalibrationReviewStatus.APPROVED,
    )

    report = _compare(references=(reference,), decisions=(decision,))

    assert report.comparisons[0].outcome is LabelCalibrationOutcome.APPROVED_ABSTAIN
    assert report.summary.approved_abstentions == 1
    assert report.summary.blocking_findings == 0


@pytest.mark.parametrize(
    ("reference", "decision", "expected_code"),
    [
        (
            _reference(),
            _decision(
                decision=LabelDecisionValueV2.NO_MATCH,
                positive=(),
                negative=(
                    _evidence(
                        "evidence-ref://calibration/negative",
                        polarity=EvidencePolarity.NEGATIVE,
                    ),
                ),
            ),
            LabelCalibrationFindingCode.DECISION_MISMATCH,
        ),
        (
            _reference(positive_ids=("evidence-ref://calibration/other",)),
            _decision(),
            LabelCalibrationFindingCode.EVIDENCE_MISMATCH,
        ),
        (
            _reference(complete=False),
            _decision(),
            LabelCalibrationFindingCode.STRUCTURED_CAPABILITY_MISMATCH,
        ),
        (
            _reference(confidence=0.8),
            _decision(),
            LabelCalibrationFindingCode.CONFIDENCE_MISMATCH,
        ),
    ],
)
def test_comparison_mismatches_produce_typed_blocking_findings(
    reference: LabelCalibrationReference,
    decision: LabelDecisionV2,
    expected_code: LabelCalibrationFindingCode,
) -> None:
    report = _compare(references=(reference,), decisions=(decision,))

    comparison = report.comparisons[0]
    assert comparison.outcome is LabelCalibrationOutcome.BLOCKING_FINDING
    assert expected_code in comparison.finding_codes
    assert report.summary.blocking_findings == 1


def test_missing_reference_and_missing_decision_are_separate_gaps() -> None:
    report = _compare(
        references=(_reference(trace_ref=_trace_ref(1), label_ref=_label_ref(1)),),
        decisions=(
            _decision(
                trace_ref=_trace_ref(2),
                label_ref=_label_ref(2),
                decision_id="label-decision://calibration/missing-reference",
            ),
        ),
    )

    outcomes = {comparison.outcome for comparison in report.comparisons}
    assert outcomes == {
        LabelCalibrationOutcome.MISSING_REFERENCE,
        LabelCalibrationOutcome.MISSING_DECISION,
    }
    assert report.summary.missing_references == 1
    assert report.summary.missing_decisions == 1


def test_missing_semantic_metadata_is_a_typed_finding_from_export_row() -> None:
    semantic = (_evidence("evidence-ref://calibration/semantic"),)
    decision = _decision(
        positive=(),
        semantic=semantic,
        confidence=0.91,
        model_profile="internal-semantic-labeler-v1",
        prompt_version="contextual-recovery-label-prompt/v1",
    )
    exporter = LabelBatchExporter()
    tracker = LabelBatchProgressTracker()
    plan = LabelBatchPlan.build(
        trace_envelope_refs=(decision.trace_envelope_ref,),
        label_spec_refs=(decision.label_spec_ref,),
        audit=_audit(),
    )
    checkpoint = tracker.initialize(plan, audit=_audit())
    checkpoint = tracker.record_decision(
        checkpoint,
        pair_id=checkpoint.pairs[0].pair_id,
        decision=decision,
        audit=_audit(),
    )
    row = exporter.rows(checkpoint, decisions=(decision,))[0]
    row["model_profile"] = None
    reference = _reference(
        positive_ids=(),
        semantic_ids=("evidence-ref://calibration/semantic",),
        semantic_status=LabelCalibrationSemanticResidualStatus.REQUIRED,
        confidence=0.91,
        model_profile="internal-semantic-labeler-v1",
        prompt_version="contextual-recovery-label-prompt/v1",
    )

    report = _compare(
        references=(reference,),
        export_rows=(row,),
        batch_export_ref=_ref(
            "label-batch-export",
            "label-batch-export://calibration/1",
            FOURTH_HASH,
        ),
    )

    assert LabelCalibrationFindingCode.MISSING_SEMANTIC_METADATA in (report.comparisons[0].finding_codes)


def test_r3_05_export_row_can_drive_exact_calibration() -> None:
    decision = _decision()
    exporter = LabelBatchExporter()
    tracker = LabelBatchProgressTracker()
    plan = LabelBatchPlan.build(
        trace_envelope_refs=(decision.trace_envelope_ref,),
        label_spec_refs=(decision.label_spec_ref,),
        audit=_audit(),
    )
    checkpoint = tracker.initialize(plan, audit=_audit())
    checkpoint = tracker.record_decision(
        checkpoint,
        pair_id=checkpoint.pairs[0].pair_id,
        decision=decision,
        audit=_audit(),
    )
    row = exporter.rows(checkpoint, decisions=(decision,))[0]

    report = _compare(
        references=(_reference(),),
        export_rows=(row,),
        batch_export_ref=_ref(
            "label-batch-export",
            "label-batch-export://calibration/1",
            FOURTH_HASH,
        ),
    )

    assert report.comparisons[0].outcome is LabelCalibrationOutcome.EXACT_MATCH


def test_report_is_explicitly_development_only_and_strict() -> None:
    report = _compare(references=(_reference(),), decisions=(_decision(),))
    payload = report.model_dump(mode="json")

    assert report.claim_scope is LabelCalibrationClaimScope.DEVELOPMENT_CANARY_ONLY
    assert report.policy_version == LABEL_CALIBRATION_POLICY_VERSION
    assert "confidence_interval" not in payload
    assert "p_value" not in payload
    assert "production" not in payload
    with pytest.raises(ValidationError):
        LabelCalibrationReference.model_validate(
            {
                **_reference().model_dump(mode="python"),
                "raw_trace": "forbidden",
            }
        )


def test_unsafe_reference_and_duplicate_pairs_fail_closed() -> None:
    for unsafe_id in (
        "final-output://forbidden",
        "secret://forbidden",
        "configured-pii://forbidden",
    ):
        with pytest.raises(ValidationError, match="unsafe"):
            _reference(positive_ids=(unsafe_id,))

    reference = _reference()
    with pytest.raises(LabelCalibrationPolicyError, match="duplicate reference"):
        _compare(references=(reference, reference), decisions=(_decision(),))


def _report_snapshot(created_at: datetime) -> tuple[str, str, tuple[str, ...]]:
    report = _compare(
        references=(_reference(),),
        decisions=(_decision(),),
        audit=_audit(created_at),
    )
    return (
        report.label_calibration_report_id,
        report.report_sha256,
        tuple(comparison.comparison_id for comparison in report.comparisons),
    )


def test_report_ids_ignore_audit_time_and_hash_seed(tmp_path: Path) -> None:
    first = _report_snapshot(datetime(2026, 7, 25, tzinfo=UTC))
    second = _report_snapshot(datetime(2026, 7, 26, tzinfo=UTC))
    script = tmp_path / "check_label_calibration_seed.py"
    test_path = Path(__file__).resolve()
    script.write_text(
        f"""
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location

spec = spec_from_file_location("label_calibration_test_module", {str(test_path)!r})
assert spec is not None and spec.loader is not None
module = module_from_spec(spec)
spec.loader.exec_module(module)
print(module._report_snapshot(datetime(2026, 7, 25, tzinfo=UTC)))
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

    assert first == second
    assert run_one.stdout == run_two.stdout
