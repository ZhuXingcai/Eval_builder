from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
    LabelDecisionValueV2,
    LabelExecutionStatus,
    LabelUnresolvedReason,
)
from eval_factory.labeling.decision import LabelDecisionRoute
from eval_factory.readiness.external_observation_models import (
    BlindLabelObservationRecordV1,
    BlindLabelObservationSetV1,
)
from eval_factory.readiness.external_reference_authoring import (
    approved_external_label_specs,
)
from eval_factory.readiness.external_reference_models import (
    ExternalReferenceAuthorKindV1,
    ExternalReferenceRecordV1,
    ExternalReferenceSetV1,
    ExternalReferenceStateV1,
)
from eval_factory.readiness.external_semantic_blocked import (
    ExternalBlockedSemanticEvidenceBuilder,
    ExternalBlockedSemanticEvidenceError,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str,
) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://external-semantic/{suffix}",
        object_version=version,
        object_sha256=_digest(f"{object_type}:{suffix}:{version}"),
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
        created_by="external-semantic-blocked-test",
        governing_versions=(
            VersionBinding(
                component="external-semantic-evidence",
                version="external-semantic-evidence/r8-10-v1",
            ),
        ),
    )


def _base_sets() -> tuple[
    ExternalReferenceSetV1,
    BlindLabelObservationSetV1,
]:
    audit = _audit()
    label_spec = next(
        spec for spec in approved_external_label_specs(audit=audit) if spec.name == "search_tool_usage"
    )
    label_ref = _ref(
        "label-spec",
        "search",
        version="v2",
    ).model_copy(
        update={
            "object_id": label_spec.label_spec_id,
            "object_sha256": label_spec.label_spec_sha256,
        }
    )
    trace_ref = _ref("trace-envelope", "trace-1", version="v1")
    source_population_ref = _ref(
        "external-source-population",
        "population",
        version="v2",
    )
    reference_policy_ref = _ref(
        "external-reference-authoring-policy",
        "reference",
        version="v2",
    )
    observation_policy_ref = _ref(
        "external-observation-policy",
        "observation",
        version="v2",
    )
    reference = ExternalReferenceRecordV1.create(
        source_trace_id="source-trace://external/trace-1",
        raw_sha256=_digest("raw-1"),
        trace_envelope_ref=trace_ref,
        label_spec_ref=label_ref,
        label_name=label_spec.name,
        annotation_contract_ref=_ref(
            "annotation-contract-manifest",
            "annotation",
            version="v1",
        ),
        reference_policy_ref=reference_policy_ref,
        expected_decision=LabelDecisionValueV2.NO_MATCH,
        evidence_refs=(trace_ref,),
        structured_capability_complete=True,
        author_kind=ExternalReferenceAuthorKindV1.DETERMINISTIC_SERVICE,
        reference_state=ExternalReferenceStateV1.REFERENCE_READY,
        rule_version="external-structured-reference/r8-10-v1",
        model_profile=None,
        prompt_version=None,
        audit=audit,
    )
    decision = LabelDecisionV2(
        label_decision_id="label-decision://external-semantic/base",
        label_spec_ref=label_ref,
        trace_envelope_ref=trace_ref,
        decision=LabelDecisionValueV2.ABSTAIN,
        execution_status=LabelExecutionStatus.UNRESOLVED,
        structured_capability_complete=True,
        confidence=0.0,
        rule_version="structured-labeling/r3-02-v1",
        unresolved_reasons=frozenset({LabelUnresolvedReason.MISSING_EVIDENCE}),
        policy_version="label-decision-merge/r3-04-v1",
        decision_sha256=_digest("base-decision"),
        audit=audit,
    )
    observation = BlindLabelObservationRecordV1.create(
        source_trace_id=reference.source_trace_id,
        raw_sha256=reference.raw_sha256,
        trace_envelope_ref=trace_ref,
        label_spec_ref=label_ref,
        observation_policy_ref=observation_policy_ref,
        label_decision=decision,
        route=LabelDecisionRoute.TYPED_UNRESOLVED_QUEUE,
        audit=audit,
    )
    return (
        ExternalReferenceSetV1.create(
            source_population_ref=source_population_ref,
            label_spec_refs=(label_ref,),
            reference_policy_ref=reference_policy_ref,
            records=(reference,),
            audit=audit,
        ),
        BlindLabelObservationSetV1.create(
            source_population_ref=source_population_ref,
            label_spec_refs=(label_ref,),
            observation_policy_ref=observation_policy_ref,
            records=(observation,),
            audit=audit,
        ),
    )


def test_blocked_semantic_evidence_extends_both_blind_authorities() -> None:
    references, observations = _base_sets()
    semantic_spec = next(
        spec
        for spec in approved_external_label_specs(audit=_audit())
        if spec.name == "contextual_recovery_after_tool_error"
    )

    result = ExternalBlockedSemanticEvidenceBuilder().extend(
        reference_set=references,
        observation_set=observations,
        semantic_label_spec=semantic_spec,
        reference_model_profile="external-semantic-reference-v1",
        reference_prompt_version=("contextual-recovery-reference/r8-10-v1"),
        audit=_audit(),
    )

    assert len(result.reference_set.records) == 2
    assert len(result.observation_set.records) == 2
    assert result.semantic_source_count == 1
    blocked_reference = next(
        record for record in result.reference_set.records if record.label_name == semantic_spec.name
    )
    blocked_observation = next(
        record
        for record in result.observation_set.records
        if record.label_spec_ref == blocked_reference.label_spec_ref
    )
    assert blocked_reference.reference_state is (ExternalReferenceStateV1.BLOCKED)
    assert blocked_reference.expected_decision is (LabelDecisionValueV2.ABSTAIN)
    assert blocked_reference.author_kind is (ExternalReferenceAuthorKindV1.SEMANTIC_AGENT)
    assert blocked_observation.label_decision.execution_status is (LabelExecutionStatus.UNRESOLVED)
    assert blocked_observation.label_decision.unresolved_reasons == (
        frozenset({LabelUnresolvedReason.MODEL_UNAVAILABLE})
    )
    assert blocked_reference.model_profile != blocked_observation.label_decision.model_profile
    assert blocked_observation.reference_access_denied is True


def test_blocked_semantic_evidence_rejects_shared_model_authority() -> None:
    references, observations = _base_sets()
    semantic_spec = next(
        spec
        for spec in approved_external_label_specs(audit=_audit())
        if spec.name == "contextual_recovery_after_tool_error"
    )
    assert semantic_spec.semantic_residual is not None

    with pytest.raises(
        ExternalBlockedSemanticEvidenceError,
        match="distinct",
    ):
        ExternalBlockedSemanticEvidenceBuilder().extend(
            reference_set=references,
            observation_set=observations,
            semantic_label_spec=semantic_spec,
            reference_model_profile=(semantic_spec.semantic_residual.model_profile),
            reference_prompt_version=(semantic_spec.semantic_residual.prompt_version),
            audit=_audit(),
        )


def test_blocked_semantic_evidence_rejects_cross_population_sets() -> None:
    references, observations = _base_sets()
    semantic_spec = next(
        spec
        for spec in approved_external_label_specs(audit=_audit())
        if spec.name == "contextual_recovery_after_tool_error"
    )
    changed = observations.model_copy(
        update={
            "source_population_ref": _ref(
                "external-source-population",
                "other",
                version="v2",
            )
        }
    )

    with pytest.raises(
        ExternalBlockedSemanticEvidenceError,
        match="population",
    ):
        ExternalBlockedSemanticEvidenceBuilder().extend(
            reference_set=references,
            observation_set=changed,
            semantic_label_spec=semantic_spec,
            reference_model_profile="external-semantic-reference-v1",
            reference_prompt_version=("contextual-recovery-reference/r8-10-v1"),
            audit=_audit(),
        )
