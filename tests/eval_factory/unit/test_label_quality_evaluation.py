from __future__ import annotations

from pathlib import Path

import pytest
from label_quality_fixtures import (
    decision,
    fixture_audit,
    frozen_authority,
    frozen_request,
    observation_set,
    quality_policy,
)

from eval_factory.contracts.label_quality_v2 import (
    LabelQualityEvidenceClassV2,
    LabelQualityMetricAvailabilityV2,
    LabelQualityOutcomeV2,
    LabelQualityReasonCodeV2,
)
from eval_factory.contracts.labeling_v2 import LabelDecisionValueV2
from eval_factory.contracts.statistics_v2 import LabelTestSetBalanceStatusV2
from eval_factory.statistics.label_quality import (
    LabelQualityEvaluator,
    wilson_score_interval,
)
from eval_factory.statistics.label_quality_builder import (
    LabelQualityEvaluationBuilder,
    LabelQualityEvaluationCompilation,
)
from eval_factory.statistics.label_quality_models import (
    LabelQualityObservationSetV1,
    LabelQualityPairResultV1,
    LabelQualityPairStatusV1,
    LabelQualityResultSetV1,
)


def _compilation(
    tmp_path: Path,
    *,
    omit_last: bool = False,
    evidence_class: LabelQualityEvidenceClassV2 = (LabelQualityEvidenceClassV2.MECHANISM_VALIDATION_ONLY),
):
    persistence, specs, frozen = frozen_authority(tmp_path)
    manifest = frozen.dataset_manifest
    assert manifest is not None
    material = persistence.material_store.get_test_set(manifest.private_material_ref)
    observations = observation_set(
        frozen,
        specs,
        material.members,
        omit_last=omit_last,
    )
    policy = quality_policy(frozen, specs, evidence_class=evidence_class)
    compilation = LabelQualityEvaluationBuilder().compile_frozen(
        persistence=persistence,
        expected_result=frozen,
        policy=policy,
        request=frozen_request(frozen, specs, observations),
        audit=fixture_audit(),
    )
    return persistence, specs, frozen, material, observations, policy, compilation


def test_perfect_frozen_evaluation_reports_all_metrics_and_passes(
    tmp_path: Path,
) -> None:
    *_unused, policy, compilation = _compilation(tmp_path)

    report = LabelQualityEvaluator().evaluate(
        compilation=compilation,
        policy=policy,
        audit=fixture_audit(),
    )

    assert report.outcome is LabelQualityOutcomeV2.PASSED
    assert report.total_reference_count == 300
    assert report.total_observation_count == 300
    assert report.missing_observation_count == 0
    assert len(report.structured_summaries) == 2
    assert len(report.semantic_summaries) == 1
    assert all(
        summary.precision.value_basis_points == 10_000 and summary.recall.value_basis_points == 10_000
        for summary in report.structured_summaries
    )
    semantic = report.semantic_summaries[0]
    assert semantic.macro_f1.value_basis_points == 10_000
    assert semantic.macro_f1_interval.lower_basis_points == 10_000
    assert semantic.macro_f1_interval.upper_basis_points == 10_000
    assert semantic.macro_f1_interval.resample_count == 10_000
    assert report.satisfies_nfr_008 is False
    assert report.satisfies_sc_010 is False


def test_evidence_class_is_policy_bound_and_cannot_be_promoted_at_evaluation(
    tmp_path: Path,
) -> None:
    *_unused, policy, compilation = _compilation(tmp_path)
    promoted = policy.model_copy(
        update={"evidence_class": LabelQualityEvidenceClassV2.PRODUCTION_INDEPENDENT_TEST}
    )

    from eval_factory.statistics.label_quality import LabelQualityEvaluationError

    with pytest.raises(LabelQualityEvaluationError, match="policy"):
        LabelQualityEvaluator().evaluate(
            compilation=compilation,
            policy=promoted,
            audit=fixture_audit(),
        )


def test_one_structured_false_positive_is_complete_failure(
    tmp_path: Path,
) -> None:
    persistence, specs, frozen, material, observations, policy, _ = _compilation(
        tmp_path,
        evidence_class=LabelQualityEvidenceClassV2.PRODUCTION_INDEPENDENT_TEST,
    )
    search = specs[1]
    target = next(
        member
        for member in material.members
        if member.label_spec_ref.object_id == search.label_spec_id
        and member.expected_decision is LabelDecisionValueV2.NO_MATCH
    )
    false_positive = decision(
        target,
        search,
        observed=LabelDecisionValueV2.MATCH,
    )
    changed = observations.create(
        dataset_manifest_ref=frozen.dataset_manifest.to_ref(),
        label_specs=specs,
        decisions=tuple(
            false_positive
            if value.trace_envelope_ref == target.trace_envelope_ref
            and value.label_spec_ref == target.label_spec_ref
            else value
            for value in observations.decisions
        ),
        audit=fixture_audit(),
    )
    compilation = LabelQualityEvaluationBuilder().compile_frozen(
        persistence=persistence,
        expected_result=frozen,
        policy=policy,
        request=frozen_request(frozen, specs, changed),
        audit=fixture_audit(),
    )

    report = LabelQualityEvaluator().evaluate(
        compilation=compilation,
        policy=policy,
        audit=fixture_audit(),
    )

    assert report.outcome is LabelQualityOutcomeV2.FAILED
    failed = next(
        summary
        for summary in report.structured_summaries
        if summary.label_spec_ref.object_id == search.label_spec_id
    )
    assert failed.true_positive_count == 50
    assert failed.false_positive_count == 1
    assert failed.precision.numerator == 50
    assert failed.precision.denominator == 51
    assert failed.precision.value_basis_points == 9_804
    assert report.satisfies_sc_010 is False


def test_missing_observation_suppresses_pass_and_remains_pending(
    tmp_path: Path,
) -> None:
    *_unused, policy, compilation = _compilation(
        tmp_path,
        omit_last=True,
        evidence_class=LabelQualityEvidenceClassV2.PRODUCTION_INDEPENDENT_TEST,
    )

    report = LabelQualityEvaluator().evaluate(
        compilation=compilation,
        policy=policy,
        audit=fixture_audit(),
    )

    assert report.outcome is LabelQualityOutcomeV2.STATISTICAL_GATE_PENDING
    assert report.missing_observation_count == 1
    assert report.satisfies_nfr_008 is False
    semantic = report.semantic_summaries[0]
    assert semantic.macro_f1.availability is LabelQualityMetricAvailabilityV2.NOT_COMPUTED
    assert semantic.macro_f1.value_basis_points is None
    assert semantic.macro_f1_interval.lower_basis_points is None
    assert semantic.macro_f1_interval.resample_count == 0


def test_missing_and_incomplete_structured_observations_preserve_both_reasons(
    tmp_path: Path,
) -> None:
    *_unused, policy, compilation = _compilation(tmp_path)
    assert compilation.result_set is not None
    target_ref = policy.structured_label_spec_refs[0]
    pairs = tuple(
        pair for pair in compilation.result_set.pair_results if pair.reference.label_spec_ref == target_ref
    )
    missing = LabelQualityPairResultV1.create(
        reference=pairs[0].reference,
        decision=None,
        status=LabelQualityPairStatusV1.MISSING,
        audit=fixture_audit(),
    )
    incomplete = LabelQualityPairResultV1.create(
        reference=pairs[1].reference,
        decision=pairs[1].decision,
        status=LabelQualityPairStatusV1.INCOMPLETE,
        audit=fixture_audit(),
    )

    summary = LabelQualityEvaluator._structured_summary(
        label_ref=target_ref,
        pairs=(missing, incomplete, *pairs[2:]),
    )

    assert summary.outcome.value == "STATISTICAL_GATE_PENDING"
    assert summary.reason_codes == (
        LabelQualityReasonCodeV2.OBSERVATION_INCOMPLETE,
        LabelQualityReasonCodeV2.OBSERVATION_MISSING,
    )
    assert summary.precision.availability is LabelQualityMetricAvailabilityV2.NOT_COMPUTED
    assert summary.recall_interval.availability is LabelQualityMetricAvailabilityV2.NOT_COMPUTED


def test_wilson_interval_is_deterministic_outward_rounded_and_undefined_safe() -> None:
    perfect = wilson_score_interval(50, 50)
    zero = wilson_score_interval(0, 50)
    undefined = wilson_score_interval(0, 0)

    assert perfect.lower_basis_points == 9_286
    assert perfect.upper_basis_points == 10_000
    assert zero.lower_basis_points == 0
    assert zero.upper_basis_points == 714
    assert undefined.availability is LabelQualityMetricAvailabilityV2.UNDEFINED
    assert undefined.lower_basis_points is None
    assert undefined.upper_basis_points is None


def test_bootstrap_and_report_identity_ignore_observation_input_order(
    tmp_path: Path,
) -> None:
    persistence, specs, frozen, _material, observations, policy, first_compilation = _compilation(tmp_path)
    reversed_observations = observations.create(
        dataset_manifest_ref=frozen.dataset_manifest.to_ref(),
        label_specs=tuple(reversed(specs)),
        decisions=tuple(reversed(observations.decisions)),
        audit=fixture_audit().model_copy(update={"created_by": "r8-02-reordered"}),
    )
    second_compilation = LabelQualityEvaluationBuilder().compile_frozen(
        persistence=persistence,
        expected_result=frozen,
        policy=policy,
        request=frozen_request(frozen, specs, reversed_observations),
        audit=fixture_audit(),
    )

    first = LabelQualityEvaluator().evaluate(
        compilation=first_compilation,
        policy=policy,
        audit=fixture_audit(),
    )
    second = LabelQualityEvaluator().evaluate(
        compilation=second_compilation,
        policy=policy,
        audit=fixture_audit().model_copy(update={"created_by": "r8-02-reordered"}),
    )

    assert first.report_sha256 == second.report_sha256
    assert first.semantic_summaries[0].macro_f1_interval == (second.semantic_summaries[0].macro_f1_interval)


def test_complete_zero_predicted_positive_is_failed_with_undefined_precision(
    tmp_path: Path,
) -> None:
    _persistence, specs, _frozen, _material, _observations, policy, compilation = _compilation(tmp_path)
    assert compilation.result_set is not None
    target_ref = policy.structured_label_spec_refs[0]
    spec = next(value for value in specs if value.label_spec_id == target_ref.object_id)
    pairs = tuple(
        LabelQualityPairResultV1.create(
            reference=pair.reference,
            decision=decision(
                pair.reference,
                spec,
                observed=LabelDecisionValueV2.NO_MATCH,
            ),
            status=LabelQualityPairStatusV1.FINAL_NO_MATCH,
            audit=fixture_audit(),
        )
        if pair.reference.expected_decision is LabelDecisionValueV2.MATCH
        else pair
        for pair in compilation.result_set.pair_results
        if pair.reference.label_spec_ref == target_ref
    )

    summary = LabelQualityEvaluator._structured_summary(
        label_ref=target_ref,
        pairs=pairs,
    )

    assert summary.precision.availability is LabelQualityMetricAvailabilityV2.UNDEFINED
    assert summary.precision.value_basis_points is None
    assert summary.reason_codes == (
        LabelQualityReasonCodeV2.METRIC_UNDEFINED,
        LabelQualityReasonCodeV2.STRUCTURED_THRESHOLD_NOT_MET,
    )


def test_source_limited_two_class_semantic_bootstrap_remains_measured(
    tmp_path: Path,
) -> None:
    _persistence, specs, _frozen, _material, _observations, policy, compilation = _compilation(tmp_path)
    assert compilation.observation_set is not None
    assert compilation.result_set is not None
    semantic_ref = policy.semantic_label_spec_ref
    semantic_spec = specs[2]
    source_pairs = tuple(
        value
        for value in compilation.result_set.pair_results
        if value.reference.label_spec_ref == semantic_ref
    )
    transformed: list[LabelQualityPairResultV1] = []
    transformed_decisions = []
    for index, pair in enumerate(source_pairs):
        expected = LabelDecisionValueV2.MATCH if index < 50 else LabelDecisionValueV2.NO_MATCH
        reference = pair.reference.model_copy(update={"expected_decision": expected})
        observed = decision(reference, semantic_spec, observed=expected)
        transformed.append(
            LabelQualityPairResultV1.create(
                reference=reference,
                decision=observed,
                status=(
                    LabelQualityPairStatusV1.FINAL_MATCH
                    if expected is LabelDecisionValueV2.MATCH
                    else LabelQualityPairStatusV1.FINAL_NO_MATCH
                ),
                audit=fixture_audit(),
            )
        )
        transformed_decisions.append(observed)
    structured_pairs = tuple(
        value
        for value in compilation.result_set.pair_results
        if value.reference.label_spec_ref != semantic_ref
    )
    structured_decisions = tuple(value.decision for value in structured_pairs if value.decision is not None)
    observations = LabelQualityObservationSetV1.create(
        dataset_manifest_ref=compilation.result_set.dataset_manifest_ref,
        label_specs=specs,
        decisions=(*structured_decisions, *transformed_decisions),
        audit=fixture_audit(),
    )
    result_set = LabelQualityResultSetV1.create(
        dataset_manifest_ref=compilation.result_set.dataset_manifest_ref,
        observation_set_ref=observations.to_ref(),
        pair_results=(*structured_pairs, *transformed),
        audit=fixture_audit(),
    )
    source_limited = LabelQualityEvaluationCompilation(
        policy=policy,
        prerequisite=compilation.prerequisite,
        observation_set=observations,
        result_set=result_set,
        evidence_sha256=compilation.evidence_sha256,
        evidence_class=policy.evidence_class,
    )

    report = LabelQualityEvaluator().evaluate(
        compilation=source_limited,
        policy=policy,
        audit=fixture_audit(),
    )

    semantic = report.semantic_summaries[0]
    assert semantic.balance_status is LabelTestSetBalanceStatusV2.SOURCE_POPULATION_LIMITED
    assert semantic.supported_class_count == 2
    assert semantic.macro_f1.value_basis_points == 10_000
    assert semantic.macro_f1_interval.lower_basis_points == 10_000
    assert semantic.macro_f1_interval.upper_basis_points == 10_000
