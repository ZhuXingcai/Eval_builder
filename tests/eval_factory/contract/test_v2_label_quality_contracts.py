from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.label_quality_v2 import (
    LABEL_QUALITY_BOOTSTRAP_REPLICATES,
    LABEL_QUALITY_POLICY_VERSION,
    LabelQualityClassMetricV2,
    LabelQualityConfidenceIntervalV2,
    LabelQualityConfusionRowV2,
    LabelQualityEvaluationPolicyV2,
    LabelQualityEvaluationReportV2,
    LabelQualityEvidenceClassV2,
    LabelQualityIntervalMethodV2,
    LabelQualityLabelOutcomeV2,
    LabelQualityMetricAvailabilityV2,
    LabelQualityMetricNameV2,
    LabelQualityOutcomeV2,
    LabelQualityPrerequisiteOutcomeV2,
    LabelQualityPrerequisiteSummaryV2,
    LabelQualityReasonCodeV2,
    SemanticLabelQualitySummaryV2,
    StructuredLabelQualitySummaryV2,
    label_quality_evaluation_report_v2_ref,
    validate_label_quality_evaluation_report_v2_identity,
)
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionValueV2,
    LabelUnresolvedReason,
)
from eval_factory.contracts.statistics_v2 import (
    LabelTestSetBalanceStatusV2,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _ref(object_type: str, suffix: str, version: str = "v2") -> ObjectRef:
    digest = _digest(f"{object_type}:{suffix}:{version}")
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://sha256/{digest}",
        object_version=version,
        object_sha256=digest,
    )


def _audit(*refs: ObjectRef, actor: str = "r8-02-contract-test") -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 3, tzinfo=UTC),
        created_by=actor,
        governing_versions=(
            VersionBinding(
                component="eval-factory-spec",
                version="approved-r8",
            ),
        ),
        input_refs=tuple(
            sorted(
                refs,
                key=lambda ref: (
                    ref.object_type,
                    ref.object_id,
                    ref.object_version,
                    ref.object_sha256,
                ),
            )
        ),
    )


def _policy(
    evidence_class: LabelQualityEvidenceClassV2 = (LabelQualityEvidenceClassV2.PRODUCTION_INDEPENDENT_TEST),
) -> LabelQualityEvaluationPolicyV2:
    test_set_policy_ref = _ref("independent-label-test-set-policy", "r8-01")
    access_policy_ref = _ref("independent-label-test-set-access-policy", "r8-01")
    structured = (
        _ref("label-spec", "powershell-error-signature"),
        _ref("label-spec", "search-tool-usage"),
    )
    semantic = _ref("label-spec", "contextual-recovery")
    pending = ObjectRef(
        object_type="label-quality-repository-evidence",
        object_id="label-quality-repository-evidence://r8-01-pending-gold",
        object_version="json/v1",
        object_sha256=("428d9d3b16eec3a31014df9e9cf17fea8a84d3ebb81c5050cd9da67f243a1a17"),
    )
    return LabelQualityEvaluationPolicyV2.create(
        test_set_policy_ref=test_set_policy_ref,
        access_policy_ref=access_policy_ref,
        structured_label_spec_refs=structured,
        semantic_label_spec_ref=semantic,
        repository_pending_evidence_ref=pending,
        evidence_class=evidence_class,
        max_members=10_000,
        max_observations=10_000,
        max_private_bytes=100_000_000,
        max_report_bytes=10_000_000,
        audit=_audit(
            test_set_policy_ref,
            access_policy_ref,
            *structured,
            semantic,
            pending,
        ),
    )


def _measured(
    metric: LabelQualityMetricNameV2,
    numerator: int,
    denominator: int,
    *,
    decision: LabelDecisionValueV2 | None = None,
) -> LabelQualityClassMetricV2:
    return LabelQualityClassMetricV2.measured(
        metric=metric,
        decision=decision,
        numerator=numerator,
        denominator=denominator,
    )


def _interval(
    *,
    method: LabelQualityIntervalMethodV2,
    sample_count: int,
    lower: int,
    upper: int,
    resample_count: int = 0,
) -> LabelQualityConfidenceIntervalV2:
    return LabelQualityConfidenceIntervalV2(
        method=method,
        confidence_basis_points=9_500,
        lower_basis_points=lower,
        upper_basis_points=upper,
        sample_count=sample_count,
        resample_count=resample_count,
        algorithm_version=(
            "wilson-score-95/v1"
            if method is LabelQualityIntervalMethodV2.WILSON_SCORE_95_V1
            else "stratified-sha256-bootstrap-95/v1"
        ),
        availability=LabelQualityMetricAvailabilityV2.MEASURED,
    )


def _structured_summary(
    label_ref: ObjectRef,
    *,
    passed: bool = True,
) -> StructuredLabelQualitySummaryV2:
    false_positive_count = 0 if passed else 1
    true_negative_count = 50 - false_positive_count
    precision = _measured(
        LabelQualityMetricNameV2.PRECISION,
        50,
        50 + false_positive_count,
        decision=LabelDecisionValueV2.MATCH,
    )
    recall = _measured(
        LabelQualityMetricNameV2.RECALL,
        50,
        50,
        decision=LabelDecisionValueV2.MATCH,
    )
    return StructuredLabelQualitySummaryV2(
        label_spec_ref=label_ref,
        reference_match_count=50,
        reference_no_match_count=50,
        true_positive_count=50,
        false_positive_count=false_positive_count,
        true_negative_count=true_negative_count,
        false_negative_count=0,
        observed_abstain_count=0,
        incomplete_observation_count=0,
        missing_observation_count=0,
        precision=precision,
        precision_interval=_interval(
            method=LabelQualityIntervalMethodV2.WILSON_SCORE_95_V1,
            sample_count=precision.denominator,
            lower=9_280,
            upper=10_000,
        ),
        recall=recall,
        recall_interval=_interval(
            method=LabelQualityIntervalMethodV2.WILSON_SCORE_95_V1,
            sample_count=recall.denominator,
            lower=9_280,
            upper=10_000,
        ),
        precision_threshold_basis_points=9_900,
        recall_threshold_basis_points=9_900,
        outcome=(LabelQualityLabelOutcomeV2.PASSED if passed else LabelQualityLabelOutcomeV2.FAILED),
        reason_codes=(
            (LabelQualityReasonCodeV2.NONE,)
            if passed
            else (LabelQualityReasonCodeV2.STRUCTURED_THRESHOLD_NOT_MET,)
        ),
    )


def _semantic_summary(
    label_ref: ObjectRef,
) -> SemanticLabelQualitySummaryV2:
    rows = (
        LabelQualityConfusionRowV2(
            reference_class=LabelDecisionValueV2.MATCH,
            predicted_match_count=34,
            predicted_no_match_count=0,
            predicted_abstain_count=0,
        ),
        LabelQualityConfusionRowV2(
            reference_class=LabelDecisionValueV2.NO_MATCH,
            predicted_match_count=0,
            predicted_no_match_count=33,
            predicted_abstain_count=0,
        ),
        LabelQualityConfusionRowV2(
            reference_class=LabelDecisionValueV2.ABSTAIN,
            predicted_match_count=0,
            predicted_no_match_count=0,
            predicted_abstain_count=33,
        ),
    )
    metrics = tuple(
        _measured(
            metric,
            2 * support if metric is LabelQualityMetricNameV2.F1 else support,
            2 * support if metric is LabelQualityMetricNameV2.F1 else support,
            decision=decision,
        )
        for decision, support in (
            (LabelDecisionValueV2.MATCH, 34),
            (LabelDecisionValueV2.NO_MATCH, 33),
            (LabelDecisionValueV2.ABSTAIN, 33),
        )
        for metric in (
            LabelQualityMetricNameV2.PRECISION,
            LabelQualityMetricNameV2.RECALL,
            LabelQualityMetricNameV2.F1,
        )
    )
    return SemanticLabelQualitySummaryV2(
        label_spec_ref=label_ref,
        model_profile="internal-semantic-labeler-v1",
        prompt_version="contextual-recovery-label-prompt/v1",
        confusion_rows=rows,
        class_metrics=metrics,
        supported_class_count=3,
        balance_status=LabelTestSetBalanceStatusV2.ACHIEVED,
        valid_prediction_abstain_count=33,
        incomplete_observation_count=0,
        model_unavailable_count=0,
        missing_observation_count=0,
        macro_f1=_measured(
            LabelQualityMetricNameV2.MACRO_F1,
            1,
            1,
        ),
        macro_f1_interval=_interval(
            method=(LabelQualityIntervalMethodV2.STRATIFIED_SHA256_BOOTSTRAP_95_V1),
            sample_count=100,
            lower=10_000,
            upper=10_000,
            resample_count=LABEL_QUALITY_BOOTSTRAP_REPLICATES,
        ),
        macro_f1_threshold_basis_points=8_500,
        outcome=LabelQualityLabelOutcomeV2.PASSED,
        reason_codes=(LabelQualityReasonCodeV2.NONE,),
    )


def _frozen_prerequisite() -> LabelQualityPrerequisiteSummaryV2:
    freeze_ref = _ref("independent-label-test-set-freeze-result", "frozen")
    dataset_ref = _ref("independent-label-test-set-manifest", "dataset")
    receipt_ref = _ref("independent-label-test-set-access-receipt", "granted")
    return LabelQualityPrerequisiteSummaryV2(
        outcome=LabelQualityPrerequisiteOutcomeV2.FROZEN,
        freeze_result_ref=freeze_ref,
        repository_evidence_ref=None,
        dataset_manifest_ref=dataset_ref,
        access_receipt_ref=receipt_ref,
        dataset_version="r8-independent-v1",
        selected_member_count=300,
        available_semantic_count=100,
        required_semantic_count=100,
        shortage_count=0,
        material_verified=True,
    )


def _report(
    *,
    evidence_class: LabelQualityEvidenceClassV2 = (LabelQualityEvidenceClassV2.PRODUCTION_INDEPENDENT_TEST),
) -> LabelQualityEvaluationReportV2:
    policy = _policy(evidence_class)
    structured = tuple(_structured_summary(ref) for ref in policy.structured_label_spec_refs)
    semantic = _semantic_summary(policy.semantic_label_spec_ref)
    return LabelQualityEvaluationReportV2.create(
        policy_ref=policy.to_ref(),
        evidence_class=evidence_class,
        prerequisite=_frozen_prerequisite(),
        observation_closure_sha256=_digest("observation-closure"),
        result_closure_sha256=_digest("result-closure"),
        structured_summaries=structured,
        semantic_summaries=(semantic,),
        audit=_audit(policy.to_ref()),
    )


def test_policy_is_strict_frozen_and_binds_canonical_rules() -> None:
    policy = _policy(LabelQualityEvidenceClassV2.REPOSITORY_PENDING_ONLY)

    assert policy.policy_version == LABEL_QUALITY_POLICY_VERSION
    assert policy.structured_precision_threshold_basis_points == 9_900
    assert policy.structured_recall_threshold_basis_points == 9_900
    assert policy.semantic_macro_f1_threshold_basis_points == 8_500
    assert policy.bootstrap_replicate_count == LABEL_QUALITY_BOOTSTRAP_REPLICATES
    assert policy.allowed_semantic_abstain_reasons == (
        LabelUnresolvedReason.AMBIGUOUS_EVIDENCE,
        LabelUnresolvedReason.CONFLICTING_EVIDENCE,
        LabelUnresolvedReason.LOW_CONFIDENCE,
        LabelUnresolvedReason.USER_INSPECTION_REQUIRED,
    )
    assert policy.to_ref().object_sha256 == policy.policy_sha256

    with pytest.raises(ValidationError):
        LabelQualityEvaluationPolicyV2.model_validate(
            {
                **policy.model_dump(mode="python"),
                "unknown": True,
            }
        )
    with pytest.raises(ValidationError):
        policy.policy_version = "changed"  # type: ignore[misc]


def test_pending_prerequisite_forbids_dataset_and_metric_authority() -> None:
    policy = _policy(LabelQualityEvidenceClassV2.REPOSITORY_PENDING_ONLY)
    pending = LabelQualityPrerequisiteSummaryV2(
        outcome=LabelQualityPrerequisiteOutcomeV2.STATISTICAL_GATE_PENDING,
        freeze_result_ref=None,
        repository_evidence_ref=policy.repository_pending_evidence_ref,
        dataset_manifest_ref=None,
        access_receipt_ref=None,
        dataset_version=None,
        selected_member_count=0,
        available_semantic_count=91,
        required_semantic_count=100,
        shortage_count=1,
        material_verified=False,
    )

    report = LabelQualityEvaluationReportV2.create(
        policy_ref=policy.to_ref(),
        evidence_class=LabelQualityEvidenceClassV2.REPOSITORY_PENDING_ONLY,
        prerequisite=pending,
        observation_closure_sha256=None,
        result_closure_sha256=None,
        structured_summaries=(),
        semantic_summaries=(),
        audit=_audit(policy.to_ref()),
    )

    assert report.outcome is LabelQualityOutcomeV2.STATISTICAL_GATE_PENDING
    assert report.reason_codes == (
        LabelQualityReasonCodeV2.FROZEN_DATASET_UNAVAILABLE,
        LabelQualityReasonCodeV2.SAMPLE_SHORTAGE,
    )
    assert report.total_reference_count == 0
    assert report.total_observation_count == 0
    assert report.satisfies_nfr_008 is False
    assert report.satisfies_sc_010 is False

    with pytest.raises(ValidationError, match="pending prerequisite"):
        LabelQualityPrerequisiteSummaryV2(
            **{
                **pending.model_dump(mode="python"),
                "dataset_manifest_ref": _ref(
                    "independent-label-test-set-manifest",
                    "forbidden",
                ),
            }
        )


def test_metric_undefined_is_explicit_and_never_zero() -> None:
    undefined = LabelQualityClassMetricV2.undefined(
        metric=LabelQualityMetricNameV2.PRECISION,
        decision=LabelDecisionValueV2.MATCH,
    )
    not_computed = LabelQualityClassMetricV2.not_computed(
        metric=LabelQualityMetricNameV2.MACRO_F1,
        decision=None,
    )

    assert undefined.availability is LabelQualityMetricAvailabilityV2.UNDEFINED
    assert undefined.numerator is None
    assert undefined.denominator == 0
    assert undefined.value_basis_points is None
    assert not_computed.availability is LabelQualityMetricAvailabilityV2.NOT_COMPUTED

    with pytest.raises(ValidationError, match="undefined metric"):
        LabelQualityClassMetricV2(
            metric=LabelQualityMetricNameV2.PRECISION,
            decision=LabelDecisionValueV2.MATCH,
            numerator=0,
            denominator=0,
            value_basis_points=0,
            availability=LabelQualityMetricAvailabilityV2.UNDEFINED,
        )


def test_structured_and_semantic_summaries_close_aggregate_counts() -> None:
    report = _report()

    assert report.total_reference_count == 300
    assert report.total_observation_count == 300
    assert report.missing_observation_count == 0
    assert report.outcome is LabelQualityOutcomeV2.PASSED
    assert report.satisfies_nfr_008 is True
    assert report.satisfies_sc_010 is True
    assert report.authorizes_sc_011 is False
    assert report.authorizes_sc_012 is False
    assert report.authorizes_sc_013 is False
    assert report.authorizes_approval is False
    assert report.authorizes_attestation is False
    assert report.authorizes_production_release is False

    semantic = report.semantic_summaries[0]
    assert tuple(row.reference_class for row in semantic.confusion_rows) == tuple(LabelDecisionValueV2)
    assert sum(row.support for row in semantic.confusion_rows) == 100
    assert len(semantic.class_metrics) == 9


def test_semantic_macro_f1_is_rebuilt_from_confusion_matrix() -> None:
    policy = _policy()
    semantic = _semantic_summary(policy.semantic_label_spec_ref)
    payload = semantic.model_dump(mode="python")
    payload.update(
        {
            "macro_f1": _measured(
                LabelQualityMetricNameV2.MACRO_F1,
                0,
                1,
            ),
            "outcome": LabelQualityLabelOutcomeV2.FAILED,
            "reason_codes": (LabelQualityReasonCodeV2.SEMANTIC_THRESHOLD_NOT_MET,),
        }
    )

    with pytest.raises(ValidationError, match="metric fraction"):
        SemanticLabelQualitySummaryV2.model_validate(payload)


def test_confidence_interval_method_is_bound_to_owning_metric() -> None:
    policy = _policy()
    structured = _structured_summary(policy.structured_label_spec_refs[0])
    structured_payload = structured.model_dump(mode="python")
    structured_payload["precision_interval"] = _interval(
        method=LabelQualityIntervalMethodV2.STRATIFIED_SHA256_BOOTSTRAP_95_V1,
        sample_count=structured.precision.denominator,
        lower=9_000,
        upper=10_000,
        resample_count=LABEL_QUALITY_BOOTSTRAP_REPLICATES,
    )
    with pytest.raises(ValidationError, match="interval method"):
        StructuredLabelQualitySummaryV2.model_validate(structured_payload)

    semantic = _semantic_summary(policy.semantic_label_spec_ref)
    semantic_payload = semantic.model_dump(mode="python")
    semantic_payload["macro_f1_interval"] = _interval(
        method=LabelQualityIntervalMethodV2.WILSON_SCORE_95_V1,
        sample_count=semantic.reference_count,
        lower=9_000,
        upper=10_000,
    )
    with pytest.raises(ValidationError, match="interval method"):
        SemanticLabelQualitySummaryV2.model_validate(semantic_payload)


def test_complete_threshold_miss_is_failed_and_synthetic_pass_is_non_authoritative() -> None:
    policy = _policy()
    failed = LabelQualityEvaluationReportV2.create(
        policy_ref=policy.to_ref(),
        evidence_class=LabelQualityEvidenceClassV2.PRODUCTION_INDEPENDENT_TEST,
        prerequisite=_frozen_prerequisite(),
        observation_closure_sha256=_digest("observations"),
        result_closure_sha256=_digest("results"),
        structured_summaries=(
            _structured_summary(policy.structured_label_spec_refs[0], passed=False),
            _structured_summary(policy.structured_label_spec_refs[1]),
        ),
        semantic_summaries=(_semantic_summary(policy.semantic_label_spec_ref),),
        audit=_audit(policy.to_ref()),
    )
    synthetic = _report(evidence_class=LabelQualityEvidenceClassV2.MECHANISM_VALIDATION_ONLY)

    assert failed.outcome is LabelQualityOutcomeV2.FAILED
    assert failed.satisfies_nfr_008 is False
    assert failed.satisfies_sc_010 is False
    assert synthetic.outcome is LabelQualityOutcomeV2.PASSED
    assert synthetic.satisfies_nfr_008 is False
    assert synthetic.satisfies_sc_010 is False


def test_report_identity_is_audit_time_independent_and_stale_safe() -> None:
    first = _report()
    second = LabelQualityEvaluationReportV2.create(
        policy_ref=first.policy_ref,
        evidence_class=first.evidence_class,
        prerequisite=first.prerequisite,
        observation_closure_sha256=first.observation_closure_sha256,
        result_closure_sha256=first.result_closure_sha256,
        structured_summaries=first.structured_summaries,
        semantic_summaries=first.semantic_summaries,
        audit=_audit(first.policy_ref, actor="another-r8-02-actor"),
    )

    assert first.report_sha256 == second.report_sha256
    assert first.report_id == second.report_id
    assert first.canonical_sha256() != second.canonical_sha256()
    assert label_quality_evaluation_report_v2_ref(first).object_sha256 == (first.report_sha256)
    validate_label_quality_evaluation_report_v2_identity(first)

    with pytest.raises(ValidationError, match="identity is stale"):
        LabelQualityEvaluationReportV2.model_validate(
            {
                **first.model_dump(mode="python"),
                "report_sha256": "f" * 64,
            }
        )


def test_public_contracts_exclude_private_membership_fields() -> None:
    schemas = (
        LabelQualityEvaluationPolicyV2.model_json_schema(),
        LabelQualityPrerequisiteSummaryV2.model_json_schema(),
        LabelQualityClassMetricV2.model_json_schema(),
        LabelQualityConfidenceIntervalV2.model_json_schema(),
        LabelQualityConfusionRowV2.model_json_schema(),
        StructuredLabelQualitySummaryV2.model_json_schema(),
        SemanticLabelQualitySummaryV2.model_json_schema(),
        LabelQualityEvaluationReportV2.model_json_schema(),
    )
    rendered = repr(schemas).casefold()
    for forbidden in (
        "source_trace_id",
        "raw_sha256",
        "annotation_ref",
        "expected_decision",
        "private_material_ref",
        "private_result_set_ref",
        "semantic_payload",
        "bootstrap_sample",
        "physical_path",
        "credential",
        "final_output",
        "grader_rule",
        "hidden_condition",
    ):
        assert forbidden not in rendered


def test_content_free_gold_binds_methods_thresholds_and_pending_evidence() -> None:
    gold = json.loads(
        (
            REPO_ROOT / "evals/golden/eval_factory/statistics/r8-02-label-quality-evaluation-v1.json"
        ).read_text()
    )

    assert gold["policy_version"] == LABEL_QUALITY_POLICY_VERSION
    assert gold["thresholds"] == {
        "structured_precision_basis_points": 9_900,
        "structured_recall_basis_points": 9_900,
        "semantic_macro_f1_basis_points": 8_500,
    }
    assert gold["intervals"]["bootstrap_replicate_count"] == 10_000
    assert gold["repository_pending"]["r8_01_evidence_sha256"] == (
        "428d9d3b16eec3a31014df9e9cf17fea8a84d3ebb81c5050cd9da67f243a1a17"
    )
    assert gold["repository_pending"]["accepted_report_sha256"] == (
        "f5b6f5f6cab238f9c7a34196a06769393bbd4da6db75424c5dc43d34a98a58a7"
    )
    assert gold["repository_pending"]["expected_outcome"] == ("STATISTICAL_GATE_PENDING")
    assert all(value is False for value in gold["all_reports_authorize"].values())
