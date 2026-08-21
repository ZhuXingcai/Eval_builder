from __future__ import annotations

import hashlib
from decimal import (
    ROUND_CEILING,
    ROUND_FLOOR,
    Decimal,
    localcontext,
)
from fractions import Fraction

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.label_quality_v2 import (
    LABEL_QUALITY_BOOTSTRAP_REPLICATES,
    LabelQualityClassMetricV2,
    LabelQualityConfidenceIntervalV2,
    LabelQualityConfusionRowV2,
    LabelQualityEvaluationPolicyV2,
    LabelQualityEvaluationReportV2,
    LabelQualityIntervalMethodV2,
    LabelQualityLabelOutcomeV2,
    LabelQualityMetricAvailabilityV2,
    LabelQualityMetricNameV2,
    LabelQualityReasonCodeV2,
    SemanticLabelQualitySummaryV2,
    StructuredLabelQualitySummaryV2,
    validate_label_quality_evaluation_policy_v2_identity,
)
from eval_factory.contracts.labeling_v2 import LabelDecisionValueV2, LabelSpecV2
from eval_factory.contracts.statistics_v2 import LabelTestSetBalanceStatusV2
from eval_factory.statistics.label_quality_builder import (
    LabelQualityEvaluationCompilation,
)
from eval_factory.statistics.label_quality_models import (
    LabelQualityPairResultV1,
    LabelQualityPairStatusV1,
)

_DECISIONS = tuple(LabelDecisionValueV2)
_WILSON_Z = Decimal("1.959963984540054")


class LabelQualityEvaluationError(RuntimeError):
    pass


class LabelQualityEvaluator:
    def evaluate(
        self,
        *,
        compilation: LabelQualityEvaluationCompilation,
        policy: LabelQualityEvaluationPolicyV2,
        audit: ContractAudit,
    ) -> LabelQualityEvaluationReportV2:
        try:
            validate_label_quality_evaluation_policy_v2_identity(policy)
        except ValueError as exc:
            raise LabelQualityEvaluationError("label quality policy is stale") from exc
        if compilation.policy != policy or compilation.evidence_class is not policy.evidence_class:
            raise LabelQualityEvaluationError("label quality compilation differs from policy")
        if compilation.result_set is None:
            if compilation.observation_set is not None:
                raise LabelQualityEvaluationError("pending compilation has an observation authority")
            return LabelQualityEvaluationReportV2.create(
                policy_ref=policy.to_ref(),
                evidence_class=policy.evidence_class,
                prerequisite=compilation.prerequisite,
                observation_closure_sha256=None,
                result_closure_sha256=None,
                structured_summaries=(),
                semantic_summaries=(),
                audit=audit,
            )
        observation_set = compilation.observation_set
        if observation_set is None or compilation.result_set.observation_set_ref != observation_set.to_ref():
            raise LabelQualityEvaluationError("frozen compilation observation closure is incomplete")
        grouped: dict[ObjectRef, list[LabelQualityPairResultV1]] = {}
        for pair in compilation.result_set.pair_results:
            grouped.setdefault(pair.reference.label_spec_ref, []).append(pair)
        structured = tuple(
            self._structured_summary(
                label_ref=label_ref,
                pairs=tuple(grouped.get(label_ref, ())),
            )
            for label_ref in policy.structured_label_spec_refs
        )
        semantic_pairs = tuple(grouped.get(policy.semantic_label_spec_ref, ()))
        semantic_spec = next(
            value
            for value in observation_set.label_specs
            if _label_spec_ref(value) == policy.semantic_label_spec_ref
        )
        residual = semantic_spec.semantic_residual
        if residual is None:
            raise LabelQualityEvaluationError("semantic label policy resolved to a structured LabelSpec")
        semantic = (
            self._semantic_summary(
                label_ref=policy.semantic_label_spec_ref,
                model_profile=residual.model_profile,
                prompt_version=residual.prompt_version,
                pairs=semantic_pairs,
                bootstrap_seed=_bootstrap_seed(
                    policy_ref=policy.to_ref(),
                    dataset_ref=compilation.result_set.dataset_manifest_ref,
                    label_ref=policy.semantic_label_spec_ref,
                    observation_set_ref=observation_set.to_ref(),
                ),
            ),
        )
        return LabelQualityEvaluationReportV2.create(
            policy_ref=policy.to_ref(),
            evidence_class=policy.evidence_class,
            prerequisite=compilation.prerequisite,
            observation_closure_sha256=observation_set.observation_set_sha256,
            result_closure_sha256=compilation.result_set.result_set_sha256,
            structured_summaries=structured,
            semantic_summaries=semantic,
            audit=audit,
        )

    @staticmethod
    def _structured_summary(
        *,
        label_ref: ObjectRef,
        pairs: tuple[LabelQualityPairResultV1, ...],
    ) -> StructuredLabelQualitySummaryV2:
        reference_match = sum(
            value.reference.expected_decision is LabelDecisionValueV2.MATCH for value in pairs
        )
        reference_no_match = sum(
            value.reference.expected_decision is LabelDecisionValueV2.NO_MATCH for value in pairs
        )
        if reference_match < 50 or reference_no_match < 50:
            raise LabelQualityEvaluationError("structured references do not meet frozen minimums")
        tp = fp = tn = fn = abstain = incomplete = missing = 0
        for pair in pairs:
            expected = pair.reference.expected_decision
            if pair.status is LabelQualityPairStatusV1.MISSING:
                missing += 1
            elif pair.status is LabelQualityPairStatusV1.ABSTAIN:
                abstain += 1
            elif pair.status in {
                LabelQualityPairStatusV1.INCOMPLETE,
                LabelQualityPairStatusV1.MODEL_UNAVAILABLE,
                LabelQualityPairStatusV1.VALID_ABSTAIN,
            }:
                incomplete += 1
            elif pair.status is LabelQualityPairStatusV1.FINAL_MATCH:
                if expected is LabelDecisionValueV2.MATCH:
                    tp += 1
                else:
                    fp += 1
            elif pair.status is LabelQualityPairStatusV1.FINAL_NO_MATCH:
                if expected is LabelDecisionValueV2.NO_MATCH:
                    tn += 1
                else:
                    fn += 1
        pending_reasons: set[LabelQualityReasonCodeV2] = set()
        if missing:
            pending_reasons.add(LabelQualityReasonCodeV2.OBSERVATION_MISSING)
        if abstain or incomplete:
            pending_reasons.add(LabelQualityReasonCodeV2.OBSERVATION_INCOMPLETE)
        reasons: tuple[LabelQualityReasonCodeV2, ...]
        if pending_reasons:
            precision = LabelQualityClassMetricV2.not_computed(
                metric=LabelQualityMetricNameV2.PRECISION,
                decision=LabelDecisionValueV2.MATCH,
            )
            recall = LabelQualityClassMetricV2.not_computed(
                metric=LabelQualityMetricNameV2.RECALL,
                decision=LabelDecisionValueV2.MATCH,
            )
            precision_interval = _empty_interval(
                LabelQualityIntervalMethodV2.WILSON_SCORE_95_V1,
                LabelQualityMetricAvailabilityV2.NOT_COMPUTED,
            )
            recall_interval = _empty_interval(
                LabelQualityIntervalMethodV2.WILSON_SCORE_95_V1,
                LabelQualityMetricAvailabilityV2.NOT_COMPUTED,
            )
            outcome = LabelQualityLabelOutcomeV2.STATISTICAL_GATE_PENDING
            reasons = tuple(sorted(pending_reasons, key=lambda value: value.value))
        else:
            precision = _metric(
                LabelQualityMetricNameV2.PRECISION,
                LabelDecisionValueV2.MATCH,
                tp,
                tp + fp,
            )
            recall = _metric(
                LabelQualityMetricNameV2.RECALL,
                LabelDecisionValueV2.MATCH,
                tp,
                tp + fn,
            )
            precision_interval = wilson_score_interval(tp, tp + fp)
            recall_interval = wilson_score_interval(tp, tp + fn)
            if not _metric_at_least(precision, 9_900) or not _metric_at_least(recall, 9_900):
                outcome = LabelQualityLabelOutcomeV2.FAILED
                failed_reasons = {LabelQualityReasonCodeV2.STRUCTURED_THRESHOLD_NOT_MET}
                if (
                    precision.availability is not LabelQualityMetricAvailabilityV2.MEASURED
                    or recall.availability is not LabelQualityMetricAvailabilityV2.MEASURED
                ):
                    failed_reasons.add(LabelQualityReasonCodeV2.METRIC_UNDEFINED)
                reasons = tuple(sorted(failed_reasons, key=lambda value: value.value))
            else:
                outcome = LabelQualityLabelOutcomeV2.PASSED
                reasons = (LabelQualityReasonCodeV2.NONE,)
        return StructuredLabelQualitySummaryV2(
            label_spec_ref=label_ref,
            reference_match_count=reference_match,
            reference_no_match_count=reference_no_match,
            true_positive_count=tp,
            false_positive_count=fp,
            true_negative_count=tn,
            false_negative_count=fn,
            observed_abstain_count=abstain,
            incomplete_observation_count=incomplete,
            missing_observation_count=missing,
            precision=precision,
            precision_interval=precision_interval,
            recall=recall,
            recall_interval=recall_interval,
            precision_threshold_basis_points=9_900,
            recall_threshold_basis_points=9_900,
            outcome=outcome,
            reason_codes=reasons,
        )

    @staticmethod
    def _semantic_summary(
        *,
        label_ref: ObjectRef,
        model_profile: str,
        prompt_version: str,
        pairs: tuple[LabelQualityPairResultV1, ...],
        bootstrap_seed: bytes,
    ) -> SemanticLabelQualitySummaryV2:
        if len(pairs) < 100:
            raise LabelQualityEvaluationError("semantic references do not meet frozen minimum")
        matrix = {expected: {observed: 0 for observed in _DECISIONS} for expected in _DECISIONS}
        reference_counts = {value: 0 for value in _DECISIONS}
        incomplete = model_unavailable = missing = 0
        complete_pairs: list[tuple[LabelDecisionValueV2, LabelDecisionValueV2]] = []
        for pair in pairs:
            expected = pair.reference.expected_decision
            reference_counts[expected] += 1
            observed = _observed_semantic_class(pair)
            if pair.status is LabelQualityPairStatusV1.MISSING:
                missing += 1
            elif pair.status is LabelQualityPairStatusV1.MODEL_UNAVAILABLE:
                model_unavailable += 1
                incomplete += 1
            elif observed is None:
                incomplete += 1
            else:
                matrix[expected][observed] += 1
                complete_pairs.append((expected, observed))
        rows = _confusion_rows(matrix)
        class_metrics = _semantic_class_metrics(rows)
        if incomplete or missing:
            macro = LabelQualityClassMetricV2.not_computed(
                metric=LabelQualityMetricNameV2.MACRO_F1,
                decision=None,
            )
            interval = _empty_interval(
                LabelQualityIntervalMethodV2.STRATIFIED_SHA256_BOOTSTRAP_95_V1,
                LabelQualityMetricAvailabilityV2.NOT_COMPUTED,
            )
            outcome = LabelQualityLabelOutcomeV2.STATISTICAL_GATE_PENDING
            reasons: set[LabelQualityReasonCodeV2] = set()
            if missing:
                reasons.add(LabelQualityReasonCodeV2.OBSERVATION_MISSING)
            if incomplete:
                reasons.add(LabelQualityReasonCodeV2.OBSERVATION_INCOMPLETE)
            if model_unavailable:
                reasons.add(LabelQualityReasonCodeV2.SEMANTIC_EXECUTION_UNAVAILABLE)
            reason_codes = tuple(sorted(reasons, key=lambda value: value.value))
        else:
            macro_fraction = _macro_f1(rows)
            macro = LabelQualityClassMetricV2.measured(
                metric=LabelQualityMetricNameV2.MACRO_F1,
                decision=None,
                numerator=macro_fraction.numerator,
                denominator=macro_fraction.denominator,
            )
            interval = _bootstrap_interval(
                complete_pairs=tuple(complete_pairs),
                seed=bootstrap_seed,
            )
            if _metric_at_least(macro, 8_500):
                outcome = LabelQualityLabelOutcomeV2.PASSED
                reason_codes = (LabelQualityReasonCodeV2.NONE,)
            else:
                outcome = LabelQualityLabelOutcomeV2.FAILED
                reason_codes = (LabelQualityReasonCodeV2.SEMANTIC_THRESHOLD_NOT_MET,)
        reference_sizes = tuple(reference_counts[value] for value in _DECISIONS)
        balance = (
            LabelTestSetBalanceStatusV2.ACHIEVED
            if max(reference_sizes) - min(reference_sizes) <= 1
            else LabelTestSetBalanceStatusV2.SOURCE_POPULATION_LIMITED
        )
        return SemanticLabelQualitySummaryV2(
            label_spec_ref=label_ref,
            model_profile=model_profile,
            prompt_version=prompt_version,
            confusion_rows=rows,
            class_metrics=class_metrics,
            supported_class_count=sum(row.support > 0 for row in rows),
            balance_status=balance,
            valid_prediction_abstain_count=sum(row.predicted_abstain_count for row in rows),
            incomplete_observation_count=incomplete,
            model_unavailable_count=model_unavailable,
            missing_observation_count=missing,
            macro_f1=macro,
            macro_f1_interval=interval,
            macro_f1_threshold_basis_points=8_500,
            outcome=outcome,
            reason_codes=reason_codes,
        )


def wilson_score_interval(
    numerator: int,
    denominator: int,
) -> LabelQualityConfidenceIntervalV2:
    if denominator == 0:
        return _empty_interval(
            LabelQualityIntervalMethodV2.WILSON_SCORE_95_V1,
            LabelQualityMetricAvailabilityV2.UNDEFINED,
        )
    if numerator < 0 or numerator > denominator:
        raise ValueError("Wilson interval fraction is invalid")
    with localcontext() as context:
        context.prec = 50
        n = Decimal(denominator)
        proportion = Decimal(numerator) / n
        z_squared = _WILSON_Z * _WILSON_Z
        center = proportion + z_squared / (Decimal(2) * n)
        radius = _WILSON_Z * (
            (proportion * (Decimal(1) - proportion) / n + z_squared / (Decimal(4) * n * n)).sqrt()
        )
        scale = Decimal(1) + z_squared / n
        lower = max(Decimal(0), (center - radius) / scale)
        upper = min(Decimal(1), (center + radius) / scale)
        lower_basis_points = int((lower * Decimal(10_000)).to_integral_value(rounding=ROUND_FLOOR))
        upper_basis_points = int((upper * Decimal(10_000)).to_integral_value(rounding=ROUND_CEILING))
    return LabelQualityConfidenceIntervalV2(
        method=LabelQualityIntervalMethodV2.WILSON_SCORE_95_V1,
        confidence_basis_points=9_500,
        lower_basis_points=lower_basis_points,
        upper_basis_points=upper_basis_points,
        sample_count=denominator,
        resample_count=0,
        algorithm_version="wilson-score-95/v1",
        availability=LabelQualityMetricAvailabilityV2.MEASURED,
    )


def _metric(
    name: LabelQualityMetricNameV2,
    decision: LabelDecisionValueV2,
    numerator: int,
    denominator: int,
) -> LabelQualityClassMetricV2:
    if denominator == 0:
        return LabelQualityClassMetricV2.undefined(
            metric=name,
            decision=decision,
        )
    return LabelQualityClassMetricV2.measured(
        metric=name,
        decision=decision,
        numerator=numerator,
        denominator=denominator,
    )


def _semantic_class_metrics(
    rows: tuple[LabelQualityConfusionRowV2, ...],
) -> tuple[LabelQualityClassMetricV2, ...]:
    supports = {row.reference_class: row.support for row in rows}
    predicted = {decision: sum(row.predicted_count(decision) for row in rows) for decision in _DECISIONS}
    diagonal = {row.reference_class: row.predicted_count(row.reference_class) for row in rows}
    result: list[LabelQualityClassMetricV2] = []
    for decision in _DECISIONS:
        if supports[decision] == 0:
            continue
        result.extend(
            (
                _metric(
                    LabelQualityMetricNameV2.PRECISION,
                    decision,
                    diagonal[decision],
                    predicted[decision],
                ),
                _metric(
                    LabelQualityMetricNameV2.RECALL,
                    decision,
                    diagonal[decision],
                    supports[decision],
                ),
                _metric(
                    LabelQualityMetricNameV2.F1,
                    decision,
                    2 * diagonal[decision],
                    supports[decision] + predicted[decision],
                ),
            )
        )
    return tuple(result)


def _macro_f1(
    rows: tuple[LabelQualityConfusionRowV2, ...],
) -> Fraction:
    supports = {row.reference_class: row.support for row in rows}
    predicted = {decision: sum(row.predicted_count(decision) for row in rows) for decision in _DECISIONS}
    diagonal = {row.reference_class: row.predicted_count(row.reference_class) for row in rows}
    values = tuple(
        Fraction(
            2 * diagonal[decision],
            supports[decision] + predicted[decision],
        )
        for decision in _DECISIONS
        if supports[decision] > 0
    )
    if not values:
        raise LabelQualityEvaluationError("semantic macro-F1 has no supported class")
    return sum(values, start=Fraction()) / len(values)


def _bootstrap_interval(
    *,
    complete_pairs: tuple[
        tuple[LabelDecisionValueV2, LabelDecisionValueV2],
        ...,
    ],
    seed: bytes,
) -> LabelQualityConfidenceIntervalV2:
    strata = {
        decision: tuple(observed for expected, observed in complete_pairs if expected is decision)
        for decision in _DECISIONS
    }
    stream = _Sha256IndexStream(seed)
    samples: list[Fraction] = []
    for _ in range(LABEL_QUALITY_BOOTSTRAP_REPLICATES):
        matrix = {expected: {observed: 0 for observed in _DECISIONS} for expected in _DECISIONS}
        for expected in _DECISIONS:
            values = strata[expected]
            for _ in range(len(values)):
                observed = values[stream.index(len(values))]
                matrix[expected][observed] += 1
        rows = _confusion_rows(matrix)
        samples.append(_macro_f1(rows))
    samples.sort()
    lower = samples[249]
    upper = samples[9_749]
    return LabelQualityConfidenceIntervalV2(
        method=LabelQualityIntervalMethodV2.STRATIFIED_SHA256_BOOTSTRAP_95_V1,
        confidence_basis_points=9_500,
        lower_basis_points=_fraction_basis_points(lower, rounding="floor"),
        upper_basis_points=_fraction_basis_points(upper, rounding="ceiling"),
        sample_count=len(complete_pairs),
        resample_count=LABEL_QUALITY_BOOTSTRAP_REPLICATES,
        algorithm_version="stratified-sha256-bootstrap-95/v1",
        availability=LabelQualityMetricAvailabilityV2.MEASURED,
    )


class _Sha256IndexStream:
    def __init__(self, seed: bytes) -> None:
        self.seed = seed
        self.counter = 0
        self.words: list[int] = []

    def index(self, size: int) -> int:
        if size <= 0:
            raise LabelQualityEvaluationError("bootstrap cannot sample an empty stratum")
        limit = (1 << 64) // size * size
        while True:
            if not self.words:
                digest = hashlib.sha256(self.seed + self.counter.to_bytes(16, "big")).digest()
                self.counter += 1
                self.words.extend(
                    int.from_bytes(digest[offset : offset + 8], "big") for offset in range(0, 32, 8)
                )
            value = self.words.pop(0)
            if value < limit:
                return value % size


def _empty_interval(
    method: LabelQualityIntervalMethodV2,
    availability: LabelQualityMetricAvailabilityV2,
) -> LabelQualityConfidenceIntervalV2:
    return LabelQualityConfidenceIntervalV2(
        method=method,
        confidence_basis_points=9_500,
        lower_basis_points=None,
        upper_basis_points=None,
        sample_count=0,
        resample_count=0,
        algorithm_version=(
            "stratified-sha256-bootstrap-95/v1"
            if method is LabelQualityIntervalMethodV2.STRATIFIED_SHA256_BOOTSTRAP_95_V1
            else "wilson-score-95/v1"
        ),
        availability=availability,
    )


def _observed_semantic_class(
    pair: LabelQualityPairResultV1,
) -> LabelDecisionValueV2 | None:
    if pair.status is LabelQualityPairStatusV1.FINAL_MATCH:
        return LabelDecisionValueV2.MATCH
    if pair.status is LabelQualityPairStatusV1.FINAL_NO_MATCH:
        return LabelDecisionValueV2.NO_MATCH
    if pair.status is LabelQualityPairStatusV1.VALID_ABSTAIN:
        return LabelDecisionValueV2.ABSTAIN
    return None


def _confusion_rows(
    matrix: dict[
        LabelDecisionValueV2,
        dict[LabelDecisionValueV2, int],
    ],
) -> tuple[
    LabelQualityConfusionRowV2,
    LabelQualityConfusionRowV2,
    LabelQualityConfusionRowV2,
]:
    def row(expected: LabelDecisionValueV2) -> LabelQualityConfusionRowV2:
        return LabelQualityConfusionRowV2(
            reference_class=expected,
            predicted_match_count=matrix[expected][LabelDecisionValueV2.MATCH],
            predicted_no_match_count=matrix[expected][LabelDecisionValueV2.NO_MATCH],
            predicted_abstain_count=matrix[expected][LabelDecisionValueV2.ABSTAIN],
        )

    return (
        row(LabelDecisionValueV2.MATCH),
        row(LabelDecisionValueV2.NO_MATCH),
        row(LabelDecisionValueV2.ABSTAIN),
    )


def _metric_at_least(
    metric: LabelQualityClassMetricV2,
    threshold_basis_points: int,
) -> bool:
    return bool(
        metric.availability is LabelQualityMetricAvailabilityV2.MEASURED
        and metric.numerator is not None
        and metric.numerator * 10_000 >= threshold_basis_points * metric.denominator
    )


def _fraction_basis_points(
    value: Fraction,
    *,
    rounding: str,
) -> int:
    numerator = value.numerator * 10_000
    if rounding == "floor":
        return numerator // value.denominator
    return (numerator + value.denominator - 1) // value.denominator


def _bootstrap_seed(
    *,
    policy_ref: ObjectRef,
    dataset_ref: ObjectRef,
    label_ref: ObjectRef,
    observation_set_ref: ObjectRef,
) -> bytes:
    return hashlib.sha256(
        "|".join(
            (
                policy_ref.object_sha256,
                dataset_ref.object_sha256,
                label_ref.object_sha256,
                observation_set_ref.object_sha256,
                "stratified-sha256-bootstrap-95/v1",
                str(LABEL_QUALITY_BOOTSTRAP_REPLICATES),
            )
        ).encode()
    ).digest()


def _label_spec_ref(value: LabelSpecV2) -> ObjectRef:
    return ObjectRef(
        object_type="label-spec",
        object_id=value.label_spec_id,
        object_version=value.label_version,
        object_sha256=value.label_spec_sha256,
    )
