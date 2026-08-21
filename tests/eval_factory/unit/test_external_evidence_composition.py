from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from label_quality_fixtures import (
    candidates,
    freeze_source,
    label_specs,
    quality_policy,
    service,
)
from test_external_evidence_store import _audit, _inventory
from test_independent_label_test_set_freeze import _candidate_pool

from eval_factory.contracts.agent_system_v2 import (
    EvaluationRequirementSpecV2,
)
from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.external_stability_v2 import (
    ExternalRealTraceStabilityReportV2,
)
from eval_factory.contracts.label_quality_v2 import (
    LabelQualityEvaluationReportV2,
    LabelQualityEvidenceClassV2,
)
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
    LabelDecisionValueV2,
    LabelExecutionStatus,
    LabelSpecV2,
)
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityCaseOutcomeV2,
    RealTraceStabilityCaseSummaryV2,
    RealTraceStabilityFaultPointV2,
    RealTraceStabilityReasonCodeV2,
)
from eval_factory.contracts.statistics_v2 import (
    IndependentLabelTestSetFreezeResultV2,
)
from eval_factory.contracts.trace import ParseQuality
from eval_factory.labeling.decision import LabelDecisionRoute
from eval_factory.readiness.external_evidence_admission import (
    ExternalCorpusAdmissionBuilder,
    ExternalCorpusInventoryCompilation,
    ExternalPartitionCompilation,
)
from eval_factory.readiness.external_evidence_composition import (
    ExternalEvidenceComposer,
    ExternalEvidenceCompositionError,
)
from eval_factory.readiness.external_evidence_requirement import (
    ExternalRequirementCompilation,
)
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
)
from eval_factory.statistics.label_quality import LabelQualityEvaluator
from eval_factory.statistics.label_quality_builder import (
    LabelQualityEvaluationBuilder,
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
        object_id=f"{object_type}://external-composition/{suffix}",
        object_version=version,
        object_sha256=_digest(f"{object_type}:{suffix}:{version}"),
    )


def _label_ref(value: LabelSpecV2) -> ObjectRef:
    return ObjectRef(
        object_type="label-spec",
        object_id=value.label_spec_id,
        object_version=value.label_version,
        object_sha256=value.label_spec_sha256,
    )


def _inputs() -> tuple[
    ExternalRequirementCompilation,
    ExternalCorpusInventoryCompilation,
    ExternalPartitionCompilation,
    ExternalReferenceSetV1,
    BlindLabelObservationSetV1,
]:
    audit = _audit()
    inventory = _inventory()
    corpus = ExternalCorpusInventoryCompilation(
        inventory=inventory,
        source_population_ref=inventory.to_source_population_ref(),
        raw_paths=tuple(
            (
                member.source_trace_id,
                Path(f"/private/{member.raw_sha256}.json"),
            )
            for member in inventory.members
        ),
    )
    partitions = ExternalCorpusAdmissionBuilder().compile_partitions(
        inventory=inventory,
        train_members=(),
        development_members=(),
        source_authority_refs=(
            inventory.to_ref(),
            inventory.to_source_population_ref(),
        ),
        audit=audit,
        minimum_test_sources=100,
    )
    label_specs = approved_external_label_specs(audit=audit)
    structured = tuple(spec for spec in label_specs if spec.semantic_residual is None)
    semantic = next(spec for spec in label_specs if spec.semantic_residual is not None)
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
    annotation_contract_ref = _ref(
        "annotation-contract-manifest",
        "annotation",
        version="v1",
    )
    references: list[ExternalReferenceRecordV1] = []
    observations: list[BlindLabelObservationRecordV1] = []
    for member in inventory.members:
        trace_ref = ObjectRef(
            object_type="trace-envelope",
            object_id=(f"trace-envelope://external-composition/{member.sid_sha256}"),
            object_version="v1",
            object_sha256=_digest(f"trace:{member.sid_sha256}"),
        )
        for label_spec in structured:
            label_ref = _label_ref(label_spec)
            references.append(
                ExternalReferenceRecordV1.create(
                    source_trace_id=member.source_trace_id,
                    raw_sha256=member.raw_sha256,
                    trace_envelope_ref=trace_ref,
                    label_spec_ref=label_ref,
                    label_name=label_spec.name,
                    annotation_contract_ref=annotation_contract_ref,
                    reference_policy_ref=reference_policy_ref,
                    expected_decision=LabelDecisionValueV2.NO_MATCH,
                    evidence_refs=(trace_ref,),
                    structured_capability_complete=True,
                    author_kind=(ExternalReferenceAuthorKindV1.DETERMINISTIC_SERVICE),
                    reference_state=(ExternalReferenceStateV1.REFERENCE_READY),
                    rule_version=("external-structured-reference/r8-10-v1"),
                    model_profile=None,
                    prompt_version=None,
                    audit=audit,
                )
            )
            evidence = EvidenceRef(
                evidence_ref_id=(
                    f"evidence-ref://external-composition/{member.sid_sha256}/{label_spec.name}"
                ),
                subject_ref=trace_ref,
                source_spans=(
                    SourceSpanRef(
                        span_id=(f"source-span://external-composition/{member.sid_sha256}/{label_spec.name}"),
                        source_trace_id=member.source_trace_id,
                        raw_sha256=member.raw_sha256,
                    ),
                ),
                polarity=EvidencePolarity.NEGATIVE,
                capability="tool_events",
                capability_complete=True,
            )
            decision_hash = _digest(f"decision:{member.sid_sha256}:{label_spec.name}")
            decision = LabelDecisionV2(
                label_decision_id=(f"label-decision://sha256/{decision_hash}"),
                label_spec_ref=label_ref,
                trace_envelope_ref=trace_ref,
                decision=LabelDecisionValueV2.NO_MATCH,
                execution_status=LabelExecutionStatus.FINAL,
                negative_evidence=(evidence,),
                structured_capability_complete=True,
                confidence=1.0,
                rule_version="structured-labeling/r3-02-v1",
                policy_version="label-decision-merge/r3-04-v1",
                decision_sha256=decision_hash,
                audit=audit,
            )
            observations.append(
                BlindLabelObservationRecordV1.create(
                    source_trace_id=member.source_trace_id,
                    raw_sha256=member.raw_sha256,
                    trace_envelope_ref=trace_ref,
                    label_spec_ref=label_ref,
                    observation_policy_ref=observation_policy_ref,
                    label_decision=decision,
                    route=LabelDecisionRoute.FINAL,
                    audit=audit,
                )
            )
    reference_set = ExternalReferenceSetV1.create(
        source_population_ref=corpus.source_population_ref,
        label_spec_refs=tuple(_label_ref(spec) for spec in structured),
        reference_policy_ref=reference_policy_ref,
        records=tuple(references),
        audit=audit,
    )
    observation_set = BlindLabelObservationSetV1.create(
        source_population_ref=corpus.source_population_ref,
        label_spec_refs=tuple(_label_ref(spec) for spec in structured),
        observation_policy_ref=observation_policy_ref,
        records=tuple(observations),
        audit=audit,
    )
    complete = ExternalBlockedSemanticEvidenceBuilder().extend(
        reference_set=reference_set,
        observation_set=observation_set,
        semantic_label_spec=semantic,
        reference_model_profile="external-semantic-reference-v1",
        reference_prompt_version=("contextual-recovery-reference/r8-10-v1"),
        audit=audit,
    )
    source_ref = _ref(
        "evaluation-requirement-source",
        "requirement",
        version="v2",
    )
    requirement_value = EvaluationRequirementSpecV2.create(
        requirement_spec_id=("evaluation-requirement-spec://external-composition/test"),
        run_id="factory-run://external-composition/test",
        source_ref=source_ref,
        goals=("Evaluate external trace labeling and stability.",),
        constraints=("Keep private evidence content-free publicly.",),
        assumptions=("The corpus is immutable.",),
        open_questions=("Semantic provider capability is unavailable.",),
        requirement_version=1,
        audit=audit,
    )
    requirement = ExternalRequirementCompilation(
        source_ref=source_ref,
        source_sha256=source_ref.object_sha256,
        source_size_bytes=100,
        media_type="text/markdown; charset=utf-8",
        requirement=requirement_value,
    )
    return (
        requirement,
        corpus,
        partitions,
        complete.reference_set,
        complete.observation_set,
    )


def test_external_package_composition_derives_exact_pending_closure() -> None:
    requirement, corpus, partitions, references, observations = _inputs()

    result = ExternalEvidenceComposer().compile_package(
        requirement=requirement,
        corpus=corpus,
        partitions=partitions,
        reference_set=references,
        observation_set=observations,
        audit=_audit(),
    )

    assert result.package_manifest.source_count == 100
    assert result.admission_report.admitted_source_count == 100
    assert result.material_closure.material_object_count == 6
    assert result.material_closure.total_pair_count == 300
    assert result.observation_summary.total_pair_count == 300
    assert result.observation_summary.reference_ready_count == 200
    assert result.observation_summary.reference_blocked_count == 100
    assert result.observation_summary.observation_complete_count == 200
    assert result.observation_summary.observation_blocked_count == 100
    assert result.observation_summary.observation_missing_count == 0


def test_external_package_composition_rejects_missing_blind_pair() -> None:
    requirement, corpus, partitions, references, observations = _inputs()
    incomplete = BlindLabelObservationSetV1.create(
        source_population_ref=observations.source_population_ref,
        label_spec_refs=observations.label_spec_refs,
        observation_policy_ref=observations.observation_policy_ref,
        records=observations.records[:-1],
        audit=_audit(),
    )

    with pytest.raises(
        ExternalEvidenceCompositionError,
        match="pair closure",
    ):
        ExternalEvidenceComposer().compile_package(
            requirement=requirement,
            corpus=corpus,
            partitions=partitions,
            reference_set=references,
            observation_set=incomplete,
            audit=_audit(),
        )


def test_external_label_material_can_bind_one_declared_umbrella_policy() -> None:
    _requirement, _corpus, _partitions, references, observations = _inputs()
    reference_policy = _ref(
        "external-reference-authoring-policy",
        "umbrella",
        version="v2",
    )
    observation_policy = _ref(
        "external-observation-policy",
        "umbrella",
        version="v2",
    )

    bound_references, bound_observations = ExternalEvidenceComposer().bind_label_material_policies(
        reference_set=references,
        observation_set=observations,
        reference_policy_ref=reference_policy,
        observation_policy_ref=observation_policy,
        audit=_audit(),
    )

    assert bound_references.reference_policy_ref == reference_policy
    assert bound_observations.observation_policy_ref == observation_policy
    assert {
        (
            record.source_trace_id,
            record.label_spec_ref,
            record.expected_decision,
        )
        for record in bound_references.records
    } == {
        (
            record.source_trace_id,
            record.label_spec_ref,
            record.expected_decision,
        )
        for record in references.records
    }
    assert {
        (
            record.source_trace_id,
            record.label_spec_ref,
            record.label_decision,
        )
        for record in bound_observations.records
    } == {
        (
            record.source_trace_id,
            record.label_spec_ref,
            record.label_decision,
        )
        for record in observations.records
    }


def _pending_label_quality(
    tmp_path: Path,
) -> tuple[
    IndependentLabelTestSetFreezeResultV2,
    LabelQualityEvaluationReportV2,
]:
    specs = label_specs()
    persistence = service(tmp_path)
    source = freeze_source(specs)
    all_candidates = candidates(specs)
    semantic_ref = all_candidates[-1].label_spec_ref
    selected = tuple(candidate for candidate in all_candidates if candidate.label_spec_ref != semantic_ref)
    selected += tuple(candidate for candidate in all_candidates if candidate.label_spec_ref == semantic_ref)[
        :91
    ]
    source["candidate_pool"] = _candidate_pool(selected)
    pending = persistence.freeze(
        idempotency_key="external-composition-pending",
        **source,  # type: ignore[arg-type]
    )
    policy = quality_policy(
        pending,
        specs,
        evidence_class=(LabelQualityEvidenceClassV2.PRODUCTION_INDEPENDENT_TEST),
    )
    compilation = LabelQualityEvaluationBuilder().compile_pending_result(
        expected_result=pending,
        policy=policy,
    )
    report = LabelQualityEvaluator().evaluate(
        compilation=compilation,
        policy=policy,
        audit=_audit(),
    )
    return pending, report


def _kernel_audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
        created_by="external-composition-stability-kernel",
        governing_versions=(
            VersionBinding(
                component="real-trace-stability",
                version="real-trace-stability/r8-04-v1",
            ),
        ),
    )


def _stable_report(
    package_ref: ObjectRef,
    inventory_ref: ObjectRef,
) -> ExternalRealTraceStabilityReportV2:
    summaries = tuple(
        RealTraceStabilityCaseSummaryV2.create(
            private_case_result_ref=_ref(
                "real-trace-stability-case-result",
                f"case-{index:03d}",
                version="private-v1",
            ),
            outcome=RealTraceStabilityCaseOutcomeV2.STABLE,
            reason_code=RealTraceStabilityReasonCodeV2.NONE,
            parse_quality=ParseQuality.STRICT,
            assigned_fault_point=tuple(RealTraceStabilityFaultPointV2)[index % 4],
            fault_observed=True,
            resume_succeeded=True,
            replay_stable=True,
            attempt_count=1,
            unexpected_retry_count=0,
            resume_count=1,
            replay_count=1,
            stage_result_count=1,
            completion_witness_count=1,
            audit=_kernel_audit(),
        )
        for index in range(100)
    )
    return ExternalRealTraceStabilityReportV2.create(
        policy_ref=_ref(
            "external-real-trace-stability-policy",
            "policy",
            version="v2",
        ),
        package_manifest_ref=package_ref,
        private_inventory_ref=inventory_ref,
        private_result_set_ref=_ref(
            "external-real-trace-stability-result-set",
            "results",
            version="private-v1",
        ),
        unique_real_trace_count=100,
        case_summaries=summaries,
        audit=_audit(),
    )


def test_external_index_composition_binds_pending_quality_and_stability(
    tmp_path: Path,
) -> None:
    requirement, corpus, partitions, references, observations = _inputs()
    package = ExternalEvidenceComposer().compile_package(
        requirement=requirement,
        corpus=corpus,
        partitions=partitions,
        reference_set=references,
        observation_set=observations,
        audit=_audit(),
    )
    freeze, quality = _pending_label_quality(tmp_path / "quality")
    stability = _stable_report(
        package.package_manifest.to_ref(),
        package.package_manifest.inventory_commitment_ref,
    )

    index = ExternalEvidenceComposer().compile_index(
        package=package,
        freeze_result=freeze,
        label_quality_report=quality,
        stability_report=stability,
        development_evidence_count=91,
        synthetic_evidence_count=0,
        audit=_audit(),
    )

    assert index.satisfies_sc_010 is False
    assert index.satisfies_sc_011 is True
    assert index.authorizes_sc_012 is False
    assert index.authorizes_attestation is False
    assert index.authorizes_production_release is False


def test_external_index_composition_rejects_cross_authority(
    tmp_path: Path,
) -> None:
    requirement, corpus, partitions, references, observations = _inputs()
    composer = ExternalEvidenceComposer()
    package = composer.compile_package(
        requirement=requirement,
        corpus=corpus,
        partitions=partitions,
        reference_set=references,
        observation_set=observations,
        audit=_audit(),
    )
    freeze, quality = _pending_label_quality(tmp_path / "quality")
    stability = _stable_report(
        package.package_manifest.to_ref(),
        package.package_manifest.inventory_commitment_ref,
    )
    changed_admission = package.admission_report.model_copy(
        update={
            "package_manifest_ref": _ref(
                "external-evidence-package-manifest",
                "other",
                version="v2",
            )
        }
    )

    with pytest.raises(
        ExternalEvidenceCompositionError,
        match="closure",
    ):
        composer.compile_index(
            package=replace(
                package,
                admission_report=changed_admission,
            ),
            freeze_result=freeze,
            label_quality_report=quality,
            stability_report=stability,
            development_evidence_count=91,
            synthetic_evidence_count=0,
            audit=_audit(),
        )

    with pytest.raises(
        ExternalEvidenceCompositionError,
        match="stability",
    ):
        composer.compile_index(
            package=package,
            freeze_result=freeze,
            label_quality_report=quality,
            stability_report=stability.model_copy(update={"unique_real_trace_count": 99}),
            development_evidence_count=91,
            synthetic_evidence_count=0,
            audit=_audit(),
        )

    with pytest.raises(
        ExternalEvidenceCompositionError,
        match="non-negative",
    ):
        composer.compile_index(
            package=package,
            freeze_result=freeze,
            label_quality_report=quality,
            stability_report=stability,
            development_evidence_count=-1,
            synthetic_evidence_count=0,
            audit=_audit(),
        )
