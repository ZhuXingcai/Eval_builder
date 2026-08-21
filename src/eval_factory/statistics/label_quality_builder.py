from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.label_quality_v2 import (
    LABEL_QUALITY_REPOSITORY_PENDING_SHA256,
    LabelQualityEvaluationPolicyV2,
    LabelQualityEvidenceClassV2,
    LabelQualityPrerequisiteOutcomeV2,
    LabelQualityPrerequisiteSummaryV2,
    validate_label_quality_evaluation_policy_v2_identity,
)
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
    LabelDecisionValueV2,
    LabelExecutionStatus,
    LabelSpecV2,
    LabelUnresolvedReason,
)
from eval_factory.contracts.statistics_v2 import (
    IndependentLabelTestSetFreezeResultV2,
    LabelTestSetEvaluationKindV2,
    LabelTestSetFreezeOutcomeV2,
)
from eval_factory.statistics.label_quality_models import (
    LabelQualityFrozenEvaluationRequestV1,
    LabelQualityObservationSetV1,
    LabelQualityPairResultV1,
    LabelQualityPairStatusV1,
    LabelQualityResultSetV1,
)
from eval_factory.statistics.models import IndependentLabelReferenceCandidateV1
from eval_factory.statistics.persistence import (
    IndependentLabelTestSetAuthorizationError,
    IndependentLabelTestSetPersistenceError,
    IndependentLabelTestSetPersistenceService,
)


class LabelQualityEvaluationBuilderError(RuntimeError):
    pass


class LabelQualityEvaluationPolicyError(LabelQualityEvaluationBuilderError):
    pass


class LabelQualityEvaluationIntegrityError(LabelQualityEvaluationBuilderError):
    pass


class LabelQualityEvaluationAuthorizationError(LabelQualityEvaluationBuilderError):
    pass


@dataclass(frozen=True, slots=True)
class LabelQualityEvaluationCompilation:
    policy: LabelQualityEvaluationPolicyV2
    prerequisite: LabelQualityPrerequisiteSummaryV2
    observation_set: LabelQualityObservationSetV1 | None
    result_set: LabelQualityResultSetV1 | None
    evidence_sha256: str
    evidence_class: LabelQualityEvidenceClassV2


class LabelQualityEvaluationBuilder:
    def compile_repository_pending(
        self,
        *,
        payload: bytes,
        policy: LabelQualityEvaluationPolicyV2,
    ) -> LabelQualityEvaluationCompilation:
        try:
            validate_label_quality_evaluation_policy_v2_identity(policy)
        except ValueError as exc:
            raise LabelQualityEvaluationPolicyError("label quality policy is stale") from exc
        if policy.evidence_class is not LabelQualityEvidenceClassV2.REPOSITORY_PENDING_ONLY:
            raise LabelQualityEvaluationPolicyError("repository pending evidence class differs from policy")
        digest = hashlib.sha256(payload).hexdigest()
        if (
            digest != LABEL_QUALITY_REPOSITORY_PENDING_SHA256
            or policy.repository_pending_evidence_ref.object_sha256 != digest
        ):
            raise LabelQualityEvaluationPolicyError(
                "repository evidence differs from approved pending authority"
            )
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LabelQualityEvaluationPolicyError("repository pending evidence is invalid") from exc
        if not isinstance(decoded, dict):
            raise LabelQualityEvaluationPolicyError("repository pending evidence is invalid")
        real = decoded.get("real_corpus")
        if (
            decoded.get("schema_version") != "eval-factory-gold/r8-01-independent-label-test-set-freeze-v1"
            or decoded.get("claim_scope") != "MECHANISM_VALIDATION_ONLY"
            or not isinstance(real, dict)
            or real.get("eligible_source_count") != 91
            or real.get("required_semantic_count") != 100
            or real.get("outcome") != "STATISTICAL_GATE_PENDING"
            or real.get("production_statistical_authority") is not False
        ):
            raise LabelQualityEvaluationPolicyError(
                "repository pending evidence is not the approved shortage"
            )
        prerequisite = LabelQualityPrerequisiteSummaryV2(
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
        return LabelQualityEvaluationCompilation(
            policy=policy,
            prerequisite=prerequisite,
            observation_set=None,
            result_set=None,
            evidence_sha256=digest,
            evidence_class=policy.evidence_class,
        )

    def compile_pending_result(
        self,
        *,
        expected_result: IndependentLabelTestSetFreezeResultV2,
        policy: LabelQualityEvaluationPolicyV2,
    ) -> LabelQualityEvaluationCompilation:
        try:
            validate_label_quality_evaluation_policy_v2_identity(policy)
            expected_ref = expected_result.to_ref()
        except ValueError as exc:
            raise LabelQualityEvaluationPolicyError("label quality pending authority is stale") from exc
        record = expected_result.freeze_record
        if (
            policy.evidence_class is LabelQualityEvidenceClassV2.REPOSITORY_PENDING_ONLY
            or record.outcome is not LabelTestSetFreezeOutcomeV2.STATISTICAL_GATE_PENDING
            or expected_result.dataset_manifest is not None
            or expected_result.policy_ref != policy.test_set_policy_ref
            or expected_result.access_policy_ref != policy.access_policy_ref
            or not record.shortages
        ):
            raise LabelQualityEvaluationPolicyError("pending result differs from R8-01 authority")
        expected_portfolio = {
            *policy.structured_label_spec_refs,
            policy.semantic_label_spec_ref,
        }
        observed_portfolio = {summary.label_spec_ref for summary in record.label_summaries}
        if observed_portfolio != expected_portfolio:
            raise LabelQualityEvaluationPolicyError("pending result portfolio differs from quality policy")
        for summary in record.label_summaries:
            expected_kind = (
                LabelTestSetEvaluationKindV2.SEMANTIC
                if summary.label_spec_ref == policy.semantic_label_spec_ref
                else LabelTestSetEvaluationKindV2.STRUCTURED
            )
            if summary.evaluation_kind is not expected_kind:
                raise LabelQualityEvaluationPolicyError(
                    "pending result label kind differs from quality policy"
                )
        semantic_summary = next(
            summary
            for summary in record.label_summaries
            if summary.label_spec_ref == policy.semantic_label_spec_ref
        )
        prerequisite = LabelQualityPrerequisiteSummaryV2(
            outcome=LabelQualityPrerequisiteOutcomeV2.STATISTICAL_GATE_PENDING,
            freeze_result_ref=expected_ref,
            repository_evidence_ref=None,
            dataset_manifest_ref=None,
            access_receipt_ref=None,
            dataset_version=None,
            selected_member_count=0,
            available_semantic_count=semantic_summary.selected_total,
            required_semantic_count=100,
            shortage_count=len(record.shortages),
            material_verified=False,
        )
        return LabelQualityEvaluationCompilation(
            policy=policy,
            prerequisite=prerequisite,
            observation_set=None,
            result_set=None,
            evidence_sha256=expected_result.result_sha256,
            evidence_class=policy.evidence_class,
        )

    def compile_frozen(
        self,
        *,
        persistence: IndependentLabelTestSetPersistenceService,
        expected_result: IndependentLabelTestSetFreezeResultV2,
        policy: LabelQualityEvaluationPolicyV2,
        request: LabelQualityFrozenEvaluationRequestV1,
        audit: ContractAudit,
    ) -> LabelQualityEvaluationCompilation:
        try:
            validate_label_quality_evaluation_policy_v2_identity(policy)
            expected_ref = expected_result.to_ref()
        except ValueError as exc:
            raise LabelQualityEvaluationPolicyError("label quality frozen authority is stale") from exc
        if policy.evidence_class is LabelQualityEvidenceClassV2.REPOSITORY_PENDING_ONLY:
            raise LabelQualityEvaluationPolicyError(
                "repository-pending policy cannot evaluate a frozen dataset"
            )
        manifest = expected_result.dataset_manifest
        if (
            expected_result.freeze_record.outcome is not LabelTestSetFreezeOutcomeV2.FROZEN
            or manifest is None
            or request.expected_freeze_result_ref != expected_ref
            or request.dataset_series_id != manifest.dataset_series_id
            or expected_result.policy_ref != policy.test_set_policy_ref
            or expected_result.access_policy_ref != policy.access_policy_ref
            or request.access_request.dataset_manifest_ref != manifest.to_ref()
            or request.access_request.access_policy_ref != policy.access_policy_ref
            or request.observation_set.dataset_manifest_ref != manifest.to_ref()
        ):
            raise LabelQualityEvaluationPolicyError("frozen request differs from R8-01 authority")
        try:
            current = persistence.get_current(request.dataset_series_id)
        except (IndependentLabelTestSetPersistenceError, ValueError) as exc:
            raise LabelQualityEvaluationIntegrityError(
                "current frozen dataset authority is unavailable"
            ) from exc
        if current != expected_result:
            raise LabelQualityEvaluationIntegrityError("requested frozen result is not current")
        self._validate_portfolio(
            request.label_specs,
            request.observation_set,
            policy,
            manifest.label_spec_refs,
        )
        if (
            len(request.observation_set.decisions) > policy.max_observations
            or len(request.observation_set.canonical_json()) > policy.max_private_bytes
        ):
            raise LabelQualityEvaluationPolicyError("label quality observations exceed policy limits")
        try:
            access = persistence.authorize_and_get_material(
                principal=request.principal.to_domain(),
                request=request.access_request,
                audit=audit,
            )
        except IndependentLabelTestSetAuthorizationError as exc:
            raise LabelQualityEvaluationAuthorizationError("R8-01 statistical access was denied") from exc
        except (IndependentLabelTestSetPersistenceError, ValueError) as exc:
            raise LabelQualityEvaluationIntegrityError("R8-01 statistical access failed") from exc
        material = access.material
        if (
            material.dataset_series_id != manifest.dataset_series_id
            or material.dataset_version != manifest.dataset_version
            or material.to_ref() != manifest.private_material_ref
            or len(material.members) != manifest.selected_member_count
            or len(material.members) > policy.max_members
            or len(material.canonical_json()) > policy.max_private_bytes
        ):
            raise LabelQualityEvaluationIntegrityError("R8-01 selected material differs from manifest")
        spec_by_ref = {_label_spec_ref(value): value for value in request.label_specs}
        references = {
            _pair_key(value.label_spec_ref, value.trace_envelope_ref): value for value in material.members
        }
        observations = {
            _pair_key(value.label_spec_ref, value.trace_envelope_ref): value
            for value in request.observation_set.decisions
        }
        extra = set(observations) - set(references)
        if extra:
            raise LabelQualityEvaluationPolicyError("observation is outside frozen test-set authority")
        pair_results = tuple(
            LabelQualityPairResultV1.create(
                reference=reference,
                decision=observations.get(key),
                status=self._classify(
                    reference=reference,
                    decision=observations.get(key),
                    spec=spec_by_ref[reference.label_spec_ref],
                    policy=policy,
                ),
                audit=audit,
            )
            for key, reference in sorted(references.items())
        )
        result_set = LabelQualityResultSetV1.create(
            dataset_manifest_ref=manifest.to_ref(),
            observation_set_ref=request.observation_set.to_ref(),
            pair_results=pair_results,
            audit=audit,
        )
        semantic_summary = next(
            value
            for value in manifest.label_summaries
            if value.label_spec_ref == policy.semantic_label_spec_ref
        )
        prerequisite = LabelQualityPrerequisiteSummaryV2(
            outcome=LabelQualityPrerequisiteOutcomeV2.FROZEN,
            freeze_result_ref=expected_ref,
            repository_evidence_ref=None,
            dataset_manifest_ref=manifest.to_ref(),
            access_receipt_ref=access.receipt.to_ref(),
            dataset_version=manifest.dataset_version,
            selected_member_count=manifest.selected_member_count,
            available_semantic_count=semantic_summary.selected_total,
            required_semantic_count=100,
            shortage_count=0,
            material_verified=access.receipt.material_verified,
        )
        return LabelQualityEvaluationCompilation(
            policy=policy,
            prerequisite=prerequisite,
            observation_set=request.observation_set,
            result_set=result_set,
            evidence_sha256=expected_result.result_sha256,
            evidence_class=policy.evidence_class,
        )

    @staticmethod
    def _validate_portfolio(
        specs: tuple[LabelSpecV2, ...],
        observations: LabelQualityObservationSetV1,
        policy: LabelQualityEvaluationPolicyV2,
        manifest_refs: tuple[ObjectRef, ...],
    ) -> None:
        refs = tuple(sorted((_label_spec_ref(value) for value in specs), key=_ref_key))
        expected = tuple(
            sorted(
                (
                    *policy.structured_label_spec_refs,
                    policy.semantic_label_spec_ref,
                ),
                key=_ref_key,
            )
        )
        observed = tuple(_label_spec_ref(value) for value in observations.label_specs)
        if refs != expected or refs != tuple(sorted(manifest_refs, key=_ref_key)) or refs != observed:
            raise LabelQualityEvaluationPolicyError("LabelSpec portfolio differs from frozen policy")
        for spec in specs:
            if spec.label_spec_sha256 != _label_spec_carried_sha256(spec):
                raise LabelQualityEvaluationPolicyError("LabelSpec behavior hash is stale")
            semantic = spec.semantic_residual is not None
            if semantic != (_label_spec_ref(spec) == policy.semantic_label_spec_ref):
                raise LabelQualityEvaluationPolicyError(
                    "LabelSpec execution kind differs from quality policy"
                )

    @staticmethod
    def _classify(
        *,
        reference: IndependentLabelReferenceCandidateV1,
        decision: LabelDecisionV2 | None,
        spec: LabelSpecV2,
        policy: LabelQualityEvaluationPolicyV2,
    ) -> LabelQualityPairStatusV1:
        if decision is None:
            return LabelQualityPairStatusV1.MISSING
        if (
            decision.label_spec_ref != reference.label_spec_ref
            or decision.trace_envelope_ref != reference.trace_envelope_ref
            or decision.label_decision_id != f"label-decision://sha256/{decision.decision_sha256}"
        ):
            raise LabelQualityEvaluationPolicyError("decision differs from frozen trace-label authority")
        if decision.rule_version != policy.structured_rule_version:
            raise LabelQualityEvaluationPolicyError("decision rule version differs from LabelSpec")
        if decision.policy_version != policy.decision_policy_version:
            raise LabelQualityEvaluationPolicyError("decision policy version differs from quality policy")
        semantic = spec.semantic_residual
        if decision.semantic_evidence and (
            semantic is None
            or decision.model_profile != semantic.model_profile
            or decision.prompt_version != semantic.prompt_version
        ):
            raise LabelQualityEvaluationPolicyError(
                "semantic observation configuration differs from LabelSpec"
            )
        if not decision.semantic_evidence and (
            decision.model_profile is not None or decision.prompt_version is not None
        ):
            raise LabelQualityEvaluationPolicyError(
                "non-semantic observation cannot carry semantic configuration"
            )
        if (
            decision.decision is LabelDecisionValueV2.MATCH
            and decision.execution_status is LabelExecutionStatus.FINAL
        ):
            return LabelQualityPairStatusV1.FINAL_MATCH
        if (
            decision.decision is LabelDecisionValueV2.NO_MATCH
            and decision.execution_status is LabelExecutionStatus.FINAL
        ):
            return LabelQualityPairStatusV1.FINAL_NO_MATCH
        if LabelUnresolvedReason.MODEL_UNAVAILABLE in decision.unresolved_reasons:
            if decision.decision is not LabelDecisionValueV2.ABSTAIN or decision.semantic_evidence:
                raise LabelQualityEvaluationPolicyError(
                    "model-unavailable observation has invalid decision evidence"
                )
            return LabelQualityPairStatusV1.MODEL_UNAVAILABLE
        if decision.decision is LabelDecisionValueV2.ABSTAIN:
            if semantic is None:
                return LabelQualityPairStatusV1.ABSTAIN
            allowed_reasons = set(policy.allowed_semantic_abstain_reasons)
            substantive_reasons = allowed_reasons - {LabelUnresolvedReason.USER_INSPECTION_REQUIRED}
            if (
                decision.semantic_evidence
                and decision.model_profile == semantic.model_profile
                and decision.prompt_version == semantic.prompt_version
                and decision.unresolved_reasons.intersection(substantive_reasons)
                and decision.unresolved_reasons.issubset(allowed_reasons)
            ):
                return LabelQualityPairStatusV1.VALID_ABSTAIN
            return LabelQualityPairStatusV1.INCOMPLETE
        return LabelQualityPairStatusV1.INCOMPLETE


def _pair_key(
    label_ref: ObjectRef,
    trace_ref: ObjectRef,
) -> tuple[
    tuple[str, str, str, str],
    tuple[str, str, str, str],
]:
    return (_ref_key(label_ref), _ref_key(trace_ref))


def _label_spec_ref(value: LabelSpecV2) -> ObjectRef:
    return ObjectRef(
        object_type="label-spec",
        object_id=value.label_spec_id,
        object_version=value.label_version,
        object_sha256=value.label_spec_sha256,
    )


def _label_spec_carried_sha256(value: LabelSpecV2) -> str:
    return hashlib.sha256(
        json.dumps(
            value.model_dump(
                mode="json",
                exclude={"label_spec_id", "label_spec_sha256", "audit"},
                exclude_none=False,
            ),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )
