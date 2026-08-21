from __future__ import annotations

import hashlib
from pathlib import Path

from test_independent_label_test_set_freeze import (
    _access_policy,
    _audit,
    _candidate,
    _candidate_pool,
    _label_ref,
    _label_spec,
    _partition,
    _policy,
    _ref,
)

from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
)
from eval_factory.contracts.label_quality_v2 import (
    LABEL_QUALITY_REPOSITORY_PENDING_SHA256,
    LabelQualityEvaluationPolicyV2,
    LabelQualityEvidenceClassV2,
)
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
    LabelDecisionValueV2,
    LabelExecutionStatus,
    LabelSpecV2,
    LabelUnresolvedReason,
)
from eval_factory.contracts.statistics_v2 import (
    LabelTestSetAccessPurposeV2,
    LabelTestSetPartitionKindV2,
)
from eval_factory.statistics.label_quality_models import (
    LabelQualityFrozenEvaluationRequestV1,
    LabelQualityObservationSetV1,
    LabelQualityTrustedPrincipalV1,
)
from eval_factory.statistics.material_store import IndependentLabelMaterialStore
from eval_factory.statistics.models import (
    IndependentLabelReferenceCandidateV1,
    IndependentLabelTestSetAccessRequestV1,
    TrustedIndependentLabelTestSetPrincipalV1,
)
from eval_factory.statistics.persistence import (
    IndependentLabelTestSetPersistenceService,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def material_store(root: Path) -> IndependentLabelMaterialStore:
    return IndependentLabelMaterialStore(
        root,
        max_candidate_pool_bytes=100_000_000,
        max_partition_bytes=10_000_000,
        max_test_set_bytes=100_000_000,
        max_members=10_000,
    )


def service(tmp_path: Path) -> IndependentLabelTestSetPersistenceService:
    return IndependentLabelTestSetPersistenceService(
        tmp_path / "r8-01.sqlite3",
        material_store=material_store(tmp_path / "r8-01-material"),
    )


def label_specs() -> tuple[LabelSpecV2, LabelSpecV2, LabelSpecV2]:
    return (
        _label_spec("powershell_error_signature", semantic=False),
        _label_spec("search_tool_usage", semantic=False),
        _label_spec("contextual_recovery_after_tool_error", semantic=True),
    )


def candidates(
    specs: tuple[LabelSpecV2, LabelSpecV2, LabelSpecV2],
) -> tuple[IndependentLabelReferenceCandidateV1, ...]:
    powershell, search, recovery = specs
    return (
        *(
            _candidate(
                powershell,
                index=index,
                decision=LabelDecisionValueV2.MATCH,
            )
            for index in range(50)
        ),
        *(
            _candidate(
                powershell,
                index=index,
                decision=LabelDecisionValueV2.NO_MATCH,
            )
            for index in range(50)
        ),
        *(
            _candidate(
                search,
                index=index,
                decision=LabelDecisionValueV2.MATCH,
            )
            for index in range(50)
        ),
        *(
            _candidate(
                search,
                index=index,
                decision=LabelDecisionValueV2.NO_MATCH,
            )
            for index in range(50)
        ),
        *(
            _candidate(
                recovery,
                index=index,
                decision=LabelDecisionValueV2.MATCH,
            )
            for index in range(34)
        ),
        *(
            _candidate(
                recovery,
                index=index,
                decision=LabelDecisionValueV2.NO_MATCH,
            )
            for index in range(33)
        ),
        *(
            _candidate(
                recovery,
                index=index,
                decision=LabelDecisionValueV2.ABSTAIN,
            )
            for index in range(33)
        ),
    )


def freeze_source(
    specs: tuple[LabelSpecV2, LabelSpecV2, LabelSpecV2],
) -> dict[str, object]:
    train_manifest, train_material = _partition(LabelTestSetPartitionKindV2.TRAIN)
    development_manifest, development_material = _partition(LabelTestSetPartitionKindV2.DEVELOPMENT)
    return {
        "dataset_series_id": "independent-label-test-set://r8-02-first-labels",
        "dataset_version": "2026-08-03.r8-02-v1",
        "label_specs": specs,
        "policy": _policy(specs),
        "access_policy": _access_policy(),
        "candidate_pool": _candidate_pool(candidates(specs)),
        "train_partition_manifest": train_manifest,
        "train_partition_material": train_material,
        "development_partition_manifest": development_manifest,
        "development_partition_material": development_material,
        "supersedes_dataset_ref": None,
        "supersedes_freeze_ref": None,
        "audit": _audit(),
    }


def frozen_authority(
    tmp_path: Path,
) -> tuple[
    IndependentLabelTestSetPersistenceService,
    tuple[LabelSpecV2, LabelSpecV2, LabelSpecV2],
    object,
]:
    specs = label_specs()
    persistence = service(tmp_path)
    frozen = persistence.freeze(
        idempotency_key="r8-02-freeze",
        **freeze_source(specs),  # type: ignore[arg-type]
    )
    assert frozen.dataset_manifest is not None
    return persistence, specs, frozen


def quality_policy(
    frozen: object,
    specs: tuple[LabelSpecV2, LabelSpecV2, LabelSpecV2],
    *,
    evidence_class: LabelQualityEvidenceClassV2 = (LabelQualityEvidenceClassV2.MECHANISM_VALIDATION_ONLY),
) -> LabelQualityEvaluationPolicyV2:
    powershell, search, recovery = specs
    pending_ref = ObjectRef(
        object_type="label-quality-repository-evidence",
        object_id="label-quality-repository-evidence://r8-01-pending-gold",
        object_version="json/v1",
        object_sha256=LABEL_QUALITY_REPOSITORY_PENDING_SHA256,
    )
    return LabelQualityEvaluationPolicyV2.create(
        test_set_policy_ref=frozen.policy_ref,  # type: ignore[attr-defined]
        access_policy_ref=frozen.access_policy_ref,  # type: ignore[attr-defined]
        structured_label_spec_refs=(
            _label_ref(powershell),
            _label_ref(search),
        ),
        semantic_label_spec_ref=_label_ref(recovery),
        repository_pending_evidence_ref=pending_ref,
        evidence_class=evidence_class,
        max_members=10_000,
        max_observations=10_000,
        max_private_bytes=100_000_000,
        max_report_bytes=10_000_000,
        audit=_audit(),
    )


def evidence(
    candidate: IndependentLabelReferenceCandidateV1,
    *,
    polarity: EvidencePolarity,
    suffix: str,
) -> EvidenceRef:
    subject = _ref("normalized-label-evidence", suffix, version="v1")
    return EvidenceRef(
        evidence_ref_id=f"evidence-ref://r8-02/{suffix}",
        subject_ref=subject,
        source_spans=(
            SourceSpanRef(
                span_id=f"source-span://r8-02/{suffix}",
                source_trace_id=candidate.source_trace_id,
                raw_sha256=candidate.raw_sha256,
            ),
        ),
        polarity=polarity,
        capability="label-evaluation",
        capability_complete=True,
    )


def decision(
    candidate: IndependentLabelReferenceCandidateV1,
    spec: LabelSpecV2,
    *,
    observed: LabelDecisionValueV2 | None = None,
    model_unavailable: bool = False,
) -> LabelDecisionV2:
    value = observed or candidate.expected_decision
    suffix = digest(f"{candidate.source_trace_id}:{spec.name}:{value.value}:{model_unavailable}")
    semantic = spec.semantic_residual is not None
    positive: tuple[EvidenceRef, ...] = ()
    negative: tuple[EvidenceRef, ...] = ()
    semantic_evidence: tuple[EvidenceRef, ...] = ()
    unresolved: frozenset[LabelUnresolvedReason] = frozenset()
    execution = LabelExecutionStatus.FINAL
    model_profile: str | None = None
    prompt_version: str | None = None
    if value is LabelDecisionValueV2.MATCH:
        target = evidence(
            candidate,
            polarity=EvidencePolarity.POSITIVE,
            suffix=f"{suffix}/positive",
        )
        if semantic:
            semantic_evidence = (target,)
            model_profile = spec.semantic_residual.model_profile
            prompt_version = spec.semantic_residual.prompt_version
        else:
            positive = (target,)
    elif value is LabelDecisionValueV2.NO_MATCH:
        negative = (
            evidence(
                candidate,
                polarity=EvidencePolarity.NEGATIVE,
                suffix=f"{suffix}/negative",
            ),
        )
        if semantic:
            semantic_evidence = (
                evidence(
                    candidate,
                    polarity=EvidencePolarity.NEGATIVE,
                    suffix=f"{suffix}/semantic",
                ),
            )
            model_profile = spec.semantic_residual.model_profile
            prompt_version = spec.semantic_residual.prompt_version
    else:
        execution = LabelExecutionStatus.UNRESOLVED
        reason = (
            LabelUnresolvedReason.MODEL_UNAVAILABLE
            if model_unavailable
            else LabelUnresolvedReason.AMBIGUOUS_EVIDENCE
        )
        unresolved = frozenset({reason})
        if semantic and not model_unavailable:
            semantic_evidence = (
                evidence(
                    candidate,
                    polarity=EvidencePolarity.UNCERTAINTY,
                    suffix=f"{suffix}/semantic-abstain",
                ),
            )
            model_profile = spec.semantic_residual.model_profile
            prompt_version = spec.semantic_residual.prompt_version
    return LabelDecisionV2(
        label_decision_id=f"label-decision://sha256/{suffix}",
        label_spec_ref=candidate.label_spec_ref,
        trace_envelope_ref=candidate.trace_envelope_ref,
        decision=value,
        execution_status=execution,
        positive_evidence=positive,
        negative_evidence=negative,
        semantic_evidence=semantic_evidence,
        structured_capability_complete=not model_unavailable,
        confidence=0.0 if value is LabelDecisionValueV2.ABSTAIN else 1.0,
        rule_version="structured-labeling/r3-02-v1",
        model_profile=model_profile,
        prompt_version=prompt_version,
        unresolved_reasons=unresolved,
        policy_version="label-decision-merge/r3-04-v1",
        decision_sha256=suffix,
        audit=_audit(),
    )


def observation_set(
    frozen: object,
    specs: tuple[LabelSpecV2, LabelSpecV2, LabelSpecV2],
    members: tuple[IndependentLabelReferenceCandidateV1, ...],
    *,
    omit_last: bool = False,
    extra_decisions: tuple[LabelDecisionV2, ...] = (),
) -> LabelQualityObservationSetV1:
    by_ref = {_label_ref(spec): spec for spec in specs}
    if omit_last:
        semantic_ref = _label_ref(specs[2])
        omitted = next(value for value in reversed(members) if value.label_spec_ref == semantic_ref)
        selected = tuple(value for value in members if value != omitted)
    else:
        selected = members
    decisions = tuple(decision(candidate, by_ref[candidate.label_spec_ref]) for candidate in selected)
    return LabelQualityObservationSetV1.create(
        dataset_manifest_ref=frozen.dataset_manifest.to_ref(),  # type: ignore[attr-defined,union-attr]
        label_specs=specs,
        decisions=(*decisions, *extra_decisions),
        audit=_audit(),
    )


def frozen_request(
    frozen: object,
    specs: tuple[LabelSpecV2, LabelSpecV2, LabelSpecV2],
    observations: LabelQualityObservationSetV1,
) -> LabelQualityFrozenEvaluationRequestV1:
    manifest = frozen.dataset_manifest  # type: ignore[attr-defined]
    assert manifest is not None
    principal = TrustedIndependentLabelTestSetPrincipalV1(
        principal_ref=_ref("statistical-principal", "evaluator", version="v1"),
        allowed_purposes=frozenset({LabelTestSetAccessPurposeV2.R8_LABEL_STATISTICAL_EVALUATION}),
        allowed_dataset_refs=frozenset({manifest.to_ref()}),
        max_members=1_000,
        audit_case_ref=None,
    )
    access_request = IndependentLabelTestSetAccessRequestV1(
        dataset_manifest_ref=manifest.to_ref(),
        access_policy_ref=frozen.access_policy_ref,  # type: ignore[attr-defined]
        purpose=LabelTestSetAccessPurposeV2.R8_LABEL_STATISTICAL_EVALUATION,
        max_members=1_000,
        idempotency_key="r8-02-access",
    )
    return LabelQualityFrozenEvaluationRequestV1.create(
        expected_freeze_result_ref=frozen.to_ref(),  # type: ignore[attr-defined]
        dataset_series_id=manifest.dataset_series_id,
        label_specs=specs,
        principal=LabelQualityTrustedPrincipalV1.from_domain(principal),
        access_request=access_request,
        observation_set=observations,
        audit=_audit(),
    )


def fixture_audit() -> ContractAudit:
    return _audit()
